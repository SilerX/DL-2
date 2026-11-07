"""Evaluador comun: todos los modelos se miden con este mismo codigo sobre predicciones COCO."""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IOU_MATCH = 0.5
MAX_DETS = 100


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _micro_curve(E):
    """Curva P/R micro a IoU 0.5 usando los emparejamientos que ya calculo COCOeval."""
    scores, tp, n_gt = [], [], 0
    for e in E.evalImgs:
        if e is None or e["aRng"] != [0, 1e10]:
            continue
        n_gt += int(np.sum(np.asarray(e["gtIgnore"]) == 0))
        scores.extend(e["dtScores"])
        tp.extend((e["dtMatches"][0] > 0).tolist())
    if not scores:
        return np.array([]), np.array([]), np.array([]), n_gt
    scores, tp = np.array(scores), np.array(tp)
    order = np.argsort(-scores)
    scores, tp = scores[order], tp[order]
    tps = np.cumsum(tp)
    precision = tps / np.arange(1, len(tp) + 1)
    recall = tps / max(n_gt, 1)
    return scores, precision, recall, n_gt


def _operating_point(scores, precision, recall, conf):
    if len(scores) == 0:
        return {"conf": conf, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # scores viene ordenado de mayor a menor
    i = int(np.argmax(f1)) if conf is None else int(np.searchsorted(-scores, -conf, side="right")) - 1
    if i < 0:
        return {"conf": conf, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "conf": float(scores[i]) if conf is None else conf,
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
    }


def _confusion(gt, preds, conf, n_classes):
    """Filas = clase real, columnas = predicha; el ultimo indice es fondo (FN o FP)."""
    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    by_image = {}
    for p in preds:
        if p["score"] >= conf:
            by_image.setdefault(p["image_id"], []).append(p)
    for image_id in gt.getImgIds():
        anns = gt.loadAnns(gt.getAnnIds(imgIds=image_id))
        dets = sorted(by_image.get(image_id, []), key=lambda d: -d["score"])
        free = set(range(len(anns)))
        for d in dets:
            best, best_iou = None, IOU_MATCH
            for j in free:
                iou = _iou(d["bbox"], anns[j]["bbox"])
                if iou >= best_iou:
                    best, best_iou = j, iou
            if best is None:
                matrix[n_classes, d["category_id"]] += 1
            else:
                free.discard(best)
                matrix[anns[best]["category_id"], d["category_id"]] += 1
        for j in free:
            matrix[anns[j]["category_id"], n_classes] += 1
    return matrix


def _iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w = min(ax1, bx1) - max(a[0], b[0])
    h = min(ay1, by1) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def evaluate(annotations, predictions, conf=None):
    """conf=None elige el umbral de mayor F1 (usar en val); en test se pasa el umbral elegido en val."""
    with _quiet():
        gt = COCO(str(annotations))
    categories = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    if not predictions:
        zero = {k: 0.0 for k in ("mAP50_95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR100")}
        return {**zero, "operating_point": _operating_point(np.array([]), None, None, conf), "per_class": {}}

    with _quiet():
        dt = gt.loadRes(predictions)
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [1, 10, MAX_DETS]
        E.evaluate()
        E.accumulate()
        E.summarize()
    s = E.stats

    # precision: [IoU, recall, clase, area, maxDets]; -1 = clase sin cajas reales en esta particion
    per_class = {}
    for k, cat in enumerate(E.params.catIds):
        all_iou = E.eval["precision"][:, :, k, 0, 2]
        at_50 = E.eval["precision"][0, :, k, 0, 2]
        n_gt = len(gt.getAnnIds(catIds=cat))
        if n_gt:
            per_class[categories[cat]] = {
                "AP50_95": float(all_iou[all_iou > -1].mean()),
                "AP50": float(at_50[at_50 > -1].mean()),
                "n_gt": n_gt,
            }

    scores, precision, recall, n_gt = _micro_curve(E)
    point = _operating_point(scores, precision, recall, conf)
    step = max(len(scores) // 300, 1)
    return {
        "mAP50_95": float(s[0]),
        "mAP50": float(s[1]),
        "mAP75": float(s[2]),
        "mAP_small": float(s[3]),
        "mAP_medium": float(s[4]),
        "mAP_large": float(s[5]),
        "AR100": float(s[8]),
        "n_images": len(gt.getImgIds()),
        "n_gt": n_gt,
        "n_classes_evaluated": len(per_class),
        "operating_point": point,
        "per_class": per_class,
        "pr_curve": {
            "precision": precision[::step].round(4).tolist(),
            "recall": recall[::step].round(4).tolist(),
        },
        "confusion": {
            "labels": [categories[c] for c in sorted(categories)] + ["fondo"],
            "matrix": _confusion(gt, predictions, point["conf"], len(categories)).tolist(),
        },
    }


def write_predictions(path, predictions):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(predictions))
