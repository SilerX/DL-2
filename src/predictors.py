"""Inferencia con una interfaz unica para YOLO y modelos PyTorch."""
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from common import image_entries
from detection_eval import MAX_DETS, write_predictions
from torch_models import build_model

EVAL_CONF = 0.001


def _multi_label_predictor():
    """predict() de ultralytics deja solo la mejor clase por caja; val() y los modelos PyTorch dejan todas."""
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils import nms, ops

    class MultiLabelPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            preds = nms.non_max_suppression(
                preds, self.args.conf, self.args.iou, multi_label=True, max_det=self.args.max_det
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            return self.construct_results(preds, img, orig_imgs, **kwargs)

    return MultiLabelPredictor


class YoloPredictor:
    def __init__(self, run):
        from ultralytics import YOLO

        self.model = YOLO(run["weights"])
        self.imgsz = run["imgsz"]
        self.num_parameters = sum(p.numel() for p in self.model.model.parameters())
        self.predictor_class = _multi_label_predictor()

    def __call__(self, path, conf):
        r = self.model.predict(
            str(path),
            predictor=self.predictor_class,
            conf=conf,
            imgsz=self.imgsz,
            max_det=MAX_DETS,
            device=0,
            verbose=False,
        )[0]
        return r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), [int(c) for c in r.boxes.cls.tolist()]


class TorchPredictor:
    def __init__(self, run):
        cfg = {**run["model"]}
        if cfg["kind"] == "faster_rcnn":
            cfg["init"] = "scratch"  # los pesos reales vienen del checkpoint
        self.model = build_model(cfg, run["nc"]).cuda()
        self.model.load_state_dict(torch.load(run["weights"], map_location="cuda", weights_only=True))
        self.model.eval()
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, path, conf):
        self.model.set_score_threshold(conf)
        with Image.open(path) as im:
            image = v2.functional.to_dtype(v2.functional.pil_to_tensor(im.convert("RGB")), torch.float32, scale=True)
        out = self.model([image.cuda()])[0]
        return out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()


def load_predictor(run):
    return YoloPredictor(run) if run["framework"] == "ultralytics" else TorchPredictor(run)


def predict_split(predictor, data_dir, split, images_dir=None, conf=EVAL_CONF):
    images, _, _ = image_entries(data_dir, split)
    folder = Path(images_dir) if images_dir else Path(data_dir) / "images" / split
    predictions = []
    for info in images:
        boxes, scores, labels = predictor(folder / info["file_name"], conf)
        for (x0, y0, x1, y1), score, label in zip(boxes, scores, labels):
            predictions.append(
                {
                    "image_id": info["id"],
                    "category_id": int(label),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(score),
                }
            )
    return predictions


def load_run(run_dir):
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


__all__ = ["load_predictor", "predict_split", "load_run", "write_predictions", "EVAL_CONF"]
