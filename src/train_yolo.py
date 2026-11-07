"""Entrena YOLO desde cero (init: scratch, modelo .yaml) o con pesos COCO (init: coco, modelo .pt).

Uso: train_yolo.py --config configs/<nombre>.yaml
train_args se pasa tal cual a ultralytics; lo que no aparezca usa su valor por defecto.
Cada elemento de phases agrega o reemplaza argumentos: una fase parte del mejor modelo de la anterior.
"""
import argparse
import csv
import os
import shutil
import time
from pathlib import Path

import torch

from common import ROOT, load_config, write_json

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "weights" / "ultralytics"))


def layer_report(trainer):
    """Lee requires_grad del modelo real que va a entrenar, por capa."""
    layers = {}
    for name, p in trainer.model.named_parameters():
        parts = name.split(".")
        index = int(parts[parts.index("model") + 1])
        layer = layers.setdefault(index, {"index": index, "params": 0, "trainable": 0})
        layer["params"] += p.numel()
        layer["trainable"] += p.numel() if p.requires_grad else 0
    types = {i: type(m).__name__ for i, m in enumerate(trainer.model.model)}
    return [{**layers[i], "type": types[i]} for i in sorted(layers)]


def read_results(path):
    with Path(path).open() as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    from ultralytics import YOLO

    init, model_name = cfg["init"], str(cfg["model"])
    if init == "scratch" and not model_name.endswith(".yaml"):
        raise ValueError("desde cero requiere un modelo .yaml (sin pesos)")
    if init == "coco" and not model_name.endswith(".pt"):
        raise ValueError("transfer learning requiere un modelo .pt (pesos COCO)")
    if init == "coco":
        weights_dir = ROOT / "weights" / "ultralytics"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(weights_dir / model_name)

    data = ROOT / cfg["data"] / "data.yaml"
    run_root = ROOT / "runs"
    run_dir = run_root / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    phases = cfg.get("phases") or [{}]

    freeze_reports, merged, best_score, best_weights, best_epoch = [], [], -1.0, None, None
    source = model_name
    epoch_offset = 0
    peak = 0.0
    start = time.perf_counter()

    for i, phase in enumerate(phases):
        phase_name = f"fase{i}"
        model = YOLO(source)
        report = {}
        model.add_callback("on_train_start", lambda trainer, r=report: r.update(layers=layer_report(trainer)))
        torch.cuda.reset_peak_memory_stats()
        model.train(
            data=str(data),
            project=str(run_dir),
            name=phase_name,
            exist_ok=True,
            pretrained=init == "coco" or i > 0,
            seed=cfg["seed"],
            device=0,
            **{**cfg["train_args"], **phase},
        )
        peak = max(peak, torch.cuda.max_memory_reserved() / 1024**2)
        freeze_reports.append({"phase": i, "layers": report["layers"]})
        write_json(run_dir / "freeze_report.json", freeze_reports)

        phase_dir = Path(model.trainer.save_dir)
        rows = read_results(phase_dir / "results.csv")
        for row in rows:
            row["epoch"] = str(int(float(row["epoch"])) + epoch_offset)
        merged.extend(rows)
        scores = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        phase_best = max(scores)
        if phase_best > best_score:
            best_score = phase_best
            best_weights = Path(model.trainer.best)
            best_epoch = int(rows[scores.index(phase_best)]["epoch"])
        epoch_offset += len(rows)
        source = str(model.trainer.best)  # la siguiente fase parte de aqui

    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    final = run_dir / "best.pt"
    shutil.copy2(best_weights, final)

    seconds = time.perf_counter() - start
    write_json(
        run_dir / "run.json",
        {
            "name": cfg["name"],
            "framework": "ultralytics",
            "init": init,
            "model": cfg["model"],
            "weights": str(final),
            "imgsz": cfg["train_args"]["imgsz"],
            "data": cfg["data"],
            "train_seconds": seconds,
            "best_epoch": best_epoch,
            "epochs_run": epoch_offset,
            "peak_gpu_mb": peak,
            "config": cfg,
        },
    )
    print(f"{cfg['name']}: {seconds / 60:.1f} min, mejor epoca {best_epoch}, pesos en {final}")


if __name__ == "__main__":
    main()
