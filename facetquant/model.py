from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


NUM_CLASSES = 2  # background, crystal


def make_model(coco_weights: str | Path | None = None, max_size: int = 1024) -> nn.Module:
    """Make the crystal-only Mask R-CNN. COCO initialization is optional."""
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=max_size,
        max_size=max_size,
        box_detections_per_img=500,
    )
    if coco_weights is not None:
        state = torch.load(coco_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, NUM_CLASSES)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, NUM_CLASSES)
    return model


def save_weights(model: nn.Module, path: str | Path, half: bool = True) -> None:
    """Store inference weights in half precision to fit GitHub's file limit."""
    state = model.state_dict()
    if half:
        state = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in state.items()}
    else:
        state = {k: v.detach().cpu() for k, v in state.items()}
    torch.save(state, path)


def load_model(weights: str | Path, device: str = "cpu", max_size: int = 1024) -> nn.Module:
    model = make_model(max_size=max_size)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model
