"""Mide VRAM y velocidad del plan de transfer learning: pesos COCO y congelamiento reales.

Uso: probe_plan.py --data data/smoke --out artifacts/gpu_probe_plan.csv
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

# (etapas del backbone entrenables, tamanio, tamanio maximo, batch)
FRCNN = [(s, size, limit, b) for s in (0, 1, 2) for size, limit in ((800, 1333), (640, 1066)) for b in (2, 4)]
# (capas congeladas, batch)
YOLO_S = [(f, b) for f in (10, 0) for b in (16, 8)]


def probe_frcnn(data, stages, size, limit, batch):
    nc = len(json.loads((data / "annotations" / "train.json").read_text())["categories"])
    cfg = {"kind": "faster_rcnn", "backbone": "resnet50_fpn", "init": "coco", "min_size": size, "max_size": limit}
    model = build_model(cfg, nc).cuda()
    model.configure_trainable({"trainable_backbone_stages": stages, "train_fpn": True, "train_rpn": True})
    model.train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
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


def probe_yolo(data, freeze, batch):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "weights" / "ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "weights" / "ultralytics" / "yolov8s.pt"))
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
            epochs=3 if batch == 16 else 2,
            imgsz=640,
            batch=batch,
            freeze=freeze,
            workers=0,
            amp=False,
            plots=False,
            val=False,
            verbose=False,
            device=0,
            save=False,
        )
    return (state["end"] - state["start"]) / state["seen"], torch.cuda.max_memory_reserved() / 1024**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/smoke")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "gpu_probe_plan.csv")
    ap.add_argument("--one", nargs="+")
    args = ap.parse_args()
    data = ROOT / args.data

    if args.one:
        kind, *values = args.one
        values = list(map(int, values))
        seconds, peak = probe_frcnn(data, *values) if kind == "frcnn" else probe_yolo(data, *values)
        print("RESULT " + json.dumps({"s_per_image": seconds, "peak_mb": peak}))
        return

    cases = [("frcnn", *c) for c in FRCNN] + [("yolo", *c) for c in YOLO_S]
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["caso", "estado", "s_por_imagen", "gpu_pico_mb"])
        for case in cases:
            label = " ".join(map(str, case))
            cmd = [sys.executable, __file__, "--data", args.data, "--one", *map(str, case)]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line:
                r = json.loads(line[7:])
                row = [label, "ok", f"{r['s_per_image']:.4f}", f"{r['peak_mb']:.0f}"]
            else:
                oom = "out of memory" in (p.stdout + p.stderr).lower()
                row = [label, "sin memoria" if oom else "error", "", ""]
                if not oom:
                    print(p.stderr[-1500:], file=sys.stderr)
            writer.writerow(row)
            f.flush()
            print(*row, flush=True)


if __name__ == "__main__":
    main()
