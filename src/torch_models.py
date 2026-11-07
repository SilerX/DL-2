"""Modelos PyTorch con la misma interfaz, para compartir train_torch.py y evaluate_run.py.

  model(images, targets) -> dict de perdidas (en modo train)
  model(images)          -> lista de {"boxes", "labels", "scores"} con labels 0..nc-1 (en modo eval)
  model.set_score_threshold(conf)
  model.configure_trainable(phase)  -> congela segun la fase; se llama al inicio de cada fase
"""
import os
from pathlib import Path

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Los pesos quedan en D: (C: tiene poco espacio libre)
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "weights" / "torch"))

BACKBONES = {
    "resnet50_fpn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
    "resnet50_fpn_v2": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    "mobilenet_v3_large_fpn": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
    ),
}


class TorchvisionFasterRCNN(nn.Module):
    """Faster R-CNN de torchvision con la cabeza reemplazada para nc clases."""

    def __init__(self, nc, backbone, init, min_size, max_size, anchor_sizes=None):
        super().__init__()
        builder, coco_weights = BACKBONES[backbone]
        if init == "coco":
            self.net = builder(weights=coco_weights, min_size=min_size, max_size=max_size)
        elif init == "scratch":
            self.net = builder(weights=None, weights_backbone=None, min_size=min_size, max_size=max_size)
        else:
            raise ValueError(f"init desconocido: {init}")
        in_features = self.net.roi_heads.box_predictor.cls_score.in_features
        self.net.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc + 1)  # +1: fondo
        self.net.roi_heads.detections_per_img = 100
        if anchor_sizes is not None:
            ratios = self.net.rpn.anchor_generator.aspect_ratios
            self.net.rpn.anchor_generator = AnchorGenerator(tuple(map(tuple, anchor_sizes)), ratios)

    def forward(self, images, targets=None):
        if self.training:
            shifted = [{**t, "labels": t["labels"] + 1} for t in targets]
            return self.net(images, shifted)
        return [{**o, "labels": o["labels"] - 1} for o in self.net(images)]

    def set_score_threshold(self, conf):
        self.net.roi_heads.score_thresh = conf

    def _backbone_stages(self):
        """De la etapa mas profunda a la mas superficial."""
        body = self.net.backbone.body
        if hasattr(body, "layer4"):
            return [
                ("layer4", [body.layer4]),
                ("layer3", [body.layer3]),
                ("layer2", [body.layer2]),
                ("layer1", [body.layer1]),
                ("conv1+bn1", [body.conv1, body.bn1]),
            ]
        return [(name, [child]) for name, child in reversed(list(body.named_children()))]

    def configure_trainable(self, phase):
        """phase: trainable_backbone_stages, train_fpn, train_rpn; el clasificador final siempre entrena."""
        stages = self._backbone_stages()
        n = phase["trainable_backbone_stages"]
        for p in self.net.backbone.body.parameters():
            p.requires_grad = False
        for _, modules in stages[:n]:
            for m in modules:
                for p in m.parameters():
                    p.requires_grad = True
        for p in self.net.backbone.fpn.parameters():
            p.requires_grad = phase["train_fpn"]
        for p in self.net.rpn.parameters():
            p.requires_grad = phase["train_rpn"]
        for p in self.net.roi_heads.parameters():
            p.requires_grad = True

    def trainability_report(self):
        body = self.net.backbone.body
        groups = {name: modules for name, modules in reversed(self._backbone_stages())}
        groups["fpn"] = [self.net.backbone.fpn]
        groups["rpn"] = [self.net.rpn]
        groups["roi_heads"] = [self.net.roi_heads]
        report = {}
        for name, modules in groups.items():
            params = [p for m in modules for p in m.parameters()]
            report[f"backbone/{name}" if any(m in body.modules() for m in modules) else name] = {
                "params": sum(p.numel() for p in params),
                "trainable": sum(p.numel() for p in params if p.requires_grad),
            }
        return report


def build_model(cfg, nc):
    kind = cfg["kind"]
    if kind == "faster_rcnn":
        return TorchvisionFasterRCNN(
            nc,
            cfg["backbone"],
            cfg["init"],
            cfg["min_size"],
            cfg["max_size"],
            cfg.get("anchor_sizes"),
        )
    if kind == "custom_cnn":
        from custom_cnn import CustomDetector

        return CustomDetector(nc=nc, **cfg["params"])
    raise ValueError(f"modelo desconocido: {kind}")
