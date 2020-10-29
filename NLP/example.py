import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import util.utils as utils
import model.base_model as base_model
from dataloader.dictionary import Dictionary
from dataloader.vqafeature import VQAFeatureDataset
from util.utils import trim_collate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1024, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, adaptive=True)
    eval_dest = VQAFeatureDataset('val', dictionary, adaptive=True)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    # Got model Here
    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1, collate_fn=trim_collate)
    eval_loader = DataLoader(eval_dest, batch_size, shuffle=True, num_workers=1, collate_fn=trim_collate)
    print(model)
    print(len(train_loader.dataset))
    """
        v: [batch size x -1 x feature size]
        b: [batch size x -1 x 6]
        q: [batch size x 14]
        a: [batch size x 3129]
    """
    for i, (v, b, q, a) in enumerate(train_loader):
        print(v.shape)
        print(b.shape)
        print(q.shape)
        print(a.shape)
        a = a.cuda()
        print(b)
        break
    result = model(v, b, q, a)
    print(result)

    opt = None
    lr_default = 1e-3
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)

    def instance_bce_with_logits(logits, labels, reduction='mean'):
        assert logits.dim() == 2

        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
        if reduction == 'mean':
            loss *= labels.size(1)
        return loss

    loss = instance_bce_with_logits(result, a)
    loss.backward()
    optim.step()
    optim.zero_grad()