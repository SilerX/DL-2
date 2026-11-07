"""Evalua una ejecucion: el umbral sale de val y se aplica a test; mide tamanio y latencia.

Uso: evaluate_run.py --run runs/<nombre> --data data/<dataset>
"""
import argparse
import time
from pathlib import Path

import torch

from common import image_entries, write_json
from detection_eval import evaluate, write_predictions
from predictors import EVAL_CONF, load_predictor, load_run, predict_split

WARMUP_IMAGES = 10
LATENCY_IMAGES = 100


def measure_latency(predictor, data_dir, split, conf):
    """Una imagen a la vez, incluye lectura, preproceso, red y postproceso."""
    images, _, _ = image_entries(data_dir, split)
    folder = Path(data_dir) / "images" / split
    paths = [folder / i["file_name"] for i in images[: WARMUP_IMAGES + LATENCY_IMAGES]]
    for p in paths[:WARMUP_IMAGES]:
        predictor(p, conf)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for p in paths[WARMUP_IMAGES:]:
        predictor(p, conf)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / max(len(paths) - WARMUP_IMAGES, 1) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()

    run = load_run(args.run)
    predictor = load_predictor(run)
    annotations = args.data / "annotations"

    val_predictions = predict_split(predictor, args.data, "val")
    write_predictions(args.run / "predictions_val.json", val_predictions)
    val_metrics = evaluate(annotations / "val.json", val_predictions)
    conf = val_metrics["operating_point"]["conf"]
    if conf is None:  # sin detecciones en val
        conf = EVAL_CONF

    test_predictions = predict_split(predictor, args.data, "test")
    write_predictions(args.run / "predictions_test.json", test_predictions)
    test_metrics = evaluate(annotations / "test.json", test_predictions, conf=conf)

    latency = measure_latency(predictor, args.data, "test", conf)
    weights_mb = Path(run["weights"]).stat().st_size / 1024**2
    write_json(
        args.run / "evaluation.json",
        {
            "name": run["name"],
            "val": val_metrics,
            "test": test_metrics,
            "conf_from_val": conf,
            "latency_ms": latency,
            "fps": 1000 / latency,
            "num_parameters": predictor.num_parameters,
            "weights_mb": weights_mb,
            "train_seconds": run["train_seconds"],
            "best_epoch": run.get("best_epoch"),
            "epochs_run": run.get("epochs_run"),
            "peak_gpu_mb": run.get("peak_gpu_mb"),
        },
    )
    print(
        f"{run['name']}: test mAP50-95={test_metrics['mAP50_95']:.4f} mAP50={test_metrics['mAP50']:.4f} "
        f"P={test_metrics['operating_point']['precision']:.3f} R={test_metrics['operating_point']['recall']:.3f} "
        f"conf={conf:.3f} latencia={latency:.1f}ms"
    )


if __name__ == "__main__":
    main()
