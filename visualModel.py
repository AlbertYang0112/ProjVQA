import torch
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from grid_feats import add_attribute_config

def getVisualModel(cfgFile):
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(cfgFile)
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.freeze()
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir="./test").resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
    return model

def visualInference(model, data):
    with torch.no_grad():
        img = model.preprocess_image(data)
        features = model.backbone(img.tensor)
        outputs = model.roi_heads.get_conv5_features(features)
    return outputs

def example():
    # Get the model
    model = getVisualModel("./visualConfigs/R-50-grid.yaml")
    print("Model:")
    print(model)

    # Randomly generate images w/ different sizes
    imgSize = np.linspace(100, 500, 20).astype(np.int)
    for s in imgSize:
        rawImg = np.random.uniform(0, 1, (3, s, s))
        data = [{"image": torch.Tensor(rawImg)}]
        outputFeature = visualInference(model, data)
        print(f"Image size: 3x{s}x{s}, feature shape (NCHW): {outputFeature.shape}")

if __name__ == '__main__':
    example()