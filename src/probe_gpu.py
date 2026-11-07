"""Mide VRAM y velocidad de combinaciones candidatas sobre el dataset de prueba.

Entrena pocas iteraciones con todos los parametros entrenables (peor caso de memoria).
Cada combinacion corre en su propio proceso para aislar los errores de memoria.
Uso: probe_gpu.py --data data/smoke --out artifacts/gpu_probe.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import ROOT, DetectionDataset, collate
from torch_models import build_model

ITERATIONS = 15
SKIP_FIRST = 3

YOLO_COMBOS = [(m, 640, b) for m in ("yolov8n", "yolo11n", "yolov8s", "yolo11s", "yolov8m") for b in (8, 16)]
YOLO_COMBOS += [(m, 960, 8) for m in ("yolov8n", "yolov8s")]
FRCNN_COMBOS = [
    (bb, size, b)
    for bb in ("mobilenet_v3_large_fpn", "resnet50_fpn", "resnet50_fpn_v2")
    for size in (640, 800)
    for b in (2, 4)
]


def probe_frcnn(data, backbone, size, batch):
    model = build_model(
        {"kind": "faster_rcnn", "backbone": backbone, "init": "scratch", "min_size": size, "max_size": size * 2},
        len(json.loads((data / "annotations" / "train.json").read_text())["categories"]),
    ).cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loader = DataLoader(DetectionDataset(data, "train"), batch_size=batch, shuffle=True, collate_fn=collate)
    torch.cuda.reset_peak_memory_stats()
    done, start = 0, None
    while done < ITERATIONS:
        for images, targets in loader:
            if done == SKIP_FIRST:
                torch.cuda.synchronize()
                start = time.perf_counter()
            loss = sum(model([i.cuda() for i in images], [{k: v.cuda() for k, v in t.items()} for t in targets]).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            done += 1
            if done == ITERATIONS:
                break
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / ((ITERATIONS - SKIP_FIRST) * batch)
    return seconds, torch.cuda.max_memory_reserved() / 1024**2


def probe_yolo(data, name, size, batch):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "weights" / "ultralytics"))
    from ultralytics import YOLO

    model = YOLO(f"{name}.yaml")
    state = {}

    def batch_start(trainer):
        state["n"] = state.get("n", 0) + 1
        if state["n"] == SKIP_FIRST + 1:
            torch.cuda.synchronize()
            state["start"] = time.perf_counter()

    def batch_end(trainer):
        if state["n"] > SKIP_FIRST:
            torch.cuda.synchronize()
            state["end"] = time.perf_counter()
            state["seen"] = (state["n"] - SKIP_FIRST) * batch

    model.add_callback("on_train_batch_start", batch_start)
    model.add_callback("on_train_batch_end", batch_end)
    with tempfile.TemporaryDirectory() as folder:
        model.train(
            data=str(data / "data.yaml"),
            project=folder,
            name="probe",
            epochs=2,
            imgsz=size,
            batch=batch,
            workers=0,
            amp=False,
            plots=False,
            val=False,
            verbose=False,
            device=0,
            pretrained=False,
            save=False,
        )
    return (state["end"] - state["start"]) / state["seen"], torch.cuda.max_memory_reserved() / 1024**2


def run_one(args):
    data = ROOT / args.data
    kind, name, size, batch = args.one[0], args.one[1], int(args.one[2]), int(args.one[3])
    seconds, peak = (probe_yolo if kind == "yolo" else probe_frcnn)(data, name, size, batch)
    print("RESULT " + json.dumps({"s_per_image": seconds, "peak_mb": peak}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/smoke")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "gpu_probe.csv")
    ap.add_argument("--one", nargs=4)
    args = ap.parse_args()
    if args.one:
        run_one(args)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combos = [("yolo", *c) for c in YOLO_COMBOS] + [("frcnn", *c) for c in FRCNN_COMBOS]
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tipo", "modelo", "resolucion", "batch", "estado", "s_por_imagen", "gpu_pico_mb"])
        for kind, name, size, batch in combos:
            cmd = [sys.executable, __file__, "--data", args.data, "--one", kind, name, str(size), str(batch)]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line:
                r = json.loads(line[7:])
                row = [kind, name, size, batch, "ok", f"{r['s_per_image']:.4f}", f"{r['peak_mb']:.0f}"]
            else:
                oom = "out of memory" in (p.stdout + p.stderr).lower()
                row = [kind, name, size, batch, "sin memoria" if oom else "error", "", ""]
                if not oom:
                    print(p.stderr[-1500:], file=sys.stderr)
            writer.writerow(row)
            f.flush()
            print(*row, flush=True)


if __name__ == "__main__":
    main()
