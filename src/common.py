"""Utilidades compartidas: configuracion, dataset COCO y registro de ejecuciones."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]


def find_nulls(node, prefix=""):
    """Un null en la configuracion es una decision de diseno que aun no se toma."""
    if node is None:
        return [prefix]
    if isinstance(node, dict):
        return [n for k, v in node.items() for n in find_nulls(v, f"{prefix}.{k}".strip("."))]
    if isinstance(node, list):
        return [n for i, v in enumerate(node) for n in find_nulls(v, f"{prefix}[{i}]")]
    return []


def load_config(path, optional=("model.anchor_sizes",)):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [n for n in find_nulls(cfg) if n not in optional]
    if missing:
        sys.exit(f"{path}: faltan decisiones de diseno: {', '.join(missing)}")
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def image_entries(data_dir, split):
    coco = json.loads((Path(data_dir) / "annotations" / f"{split}.json").read_text())
    return coco["images"], coco["annotations"], coco["categories"]


def build_augmentation(aug):
    """aug viene del YAML; {} significa sin aumento."""
    steps = []
    if aug.get("zoom_out_p"):
        steps.append(v2.RandomZoomOut(fill=0, side_range=tuple(aug["zoom_out_range"]), p=aug["zoom_out_p"]))
    if aug.get("iou_crop_p"):
        steps.append(v2.RandomApply([v2.RandomIoUCrop()], p=aug["iou_crop_p"]))
    if aug.get("hflip_p"):
        steps.append(v2.RandomHorizontalFlip(aug["hflip_p"]))
    if aug.get("vflip_p"):
        steps.append(v2.RandomVerticalFlip(aug["vflip_p"]))
    if aug.get("color_jitter"):
        steps.append(v2.ColorJitter(*aug["color_jitter"]))
    steps.append(v2.SanitizeBoundingBoxes())
    return v2.Compose(steps)


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, augmentation=None):
        images, annotations, _ = image_entries(data_dir, split)
        self.images = images
        self.folder = Path(data_dir) / "images" / split
        self.augmentation = augmentation
        self.by_image = {}
        for a in annotations:
            self.by_image.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        with Image.open(self.folder / info["file_name"]) as im:
            image = v2.functional.pil_to_tensor(im.convert("RGB"))
        anns = self.by_image.get(info["id"], [])
        boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(info["height"], info["width"])),
            "labels": torch.tensor([a["category_id"] for a in anns], dtype=torch.int64),
        }
        if self.augmentation is not None:
            image, target = self.augmentation(image, target)
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
        return image, {"boxes": target["boxes"].as_subclass(torch.Tensor), "labels": target["labels"]}


def collate(batch):
    return tuple(zip(*batch))
