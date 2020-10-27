from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import util.utils as utils
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
import tools.compute_softscore
import itertools
import re


COUNTING_ONLY = False


def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


def _create_entry(img, question, answer):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' % \
        (name + '2014' if 'test'!=name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    if 'test'!=name[:4]: # train, val
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id], question, answer))
    else: # test2015
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id], question, None))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        h5_path = os.path.join(dataroot, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))

        print('loading features from h5 file')
        # with h5py.File(h5_path, 'r') as hf:
        #     self.features = np.array(hf.get('image_features'))
        #     self.spatials = np.array(hf.get('spatial_features'))
        #     if self.adaptive:
        #         self.pos_boxes = np.array(hf.get('pos_boxes'))
        hf = h5py.File(h5_path, 'r')
        self.features = hf.get('image_features')
        self.spatials = hf.get('spatial_features')
        if self.adaptive:
            self.pos_boxes = hf.get('pos_boxes')

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)
        self.tokenize()
        self.tensorize()
        # self.v_dim = self.features.size(1 if self.adaptive else 2)
        # self.s_dim = self.spatials.size(1 if self.adaptive else 2)
        self.v_dim = self.features.shape[1 if self.adaptive else 2]
        self.s_dim = self.spatials.shape[1 if self.adaptive else 2]

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.features = torch.from_numpy(self.features)
        # self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            # features = self.features[entry['image']]
            # spatials = self.spatials[entry['image']]
            features = torch.from_numpy(self.features[entry['image']])
            spatials = torch.from_numpy(self.spatials[entry['image']])
        else:
            # features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1],:]
            # spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1],:]
            features = torch.from_numpy(
                self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :])
            spatials = torch.from_numpy(
                self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :])

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, question, target
        else:
            return features, spatials, question, question_id

    def __len__(self):
        return len(self.entries)