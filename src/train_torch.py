"""Entrena Faster R-CNN o la CNN propia.

Uso: train_torch.py --config configs/<nombre>.yaml
Cada fase define su propio congelamiento y optimizador, por lo que permite
entrenar primero la cabeza y luego hacer fine-tuning.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import ROOT, DetectionDataset, build_augmentation, collate, image_entries, load_config, set_seed, write_json
from detection_eval import evaluate
from predictors import EVAL_CONF
from torch_models import build_model


def build_optimizer(model, phase):
    params = [p for p in model.parameters() if p.requires_grad]
    if phase["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=phase["lr"], momentum=phase["momentum"], weight_decay=phase["weight_decay"])
    if phase["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=phase["lr"], weight_decay=phase["weight_decay"])
    raise ValueError(f"optimizador desconocido: {phase['optimizer']}")


def build_scheduler(optimizer, phase, steps_per_epoch):
    kind = phase["scheduler"]
    if kind == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, phase["epochs"] * steps_per_epoch)
    if kind == "multistep":
        milestones = [m * steps_per_epoch for m in phase["milestones"]]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=phase["gamma"])
    raise ValueError(f"scheduler desconocido: {kind}")


@torch.inference_mode()
def predict_val(model, loader, images):
    model.eval()
    model.set_score_threshold(EVAL_CONF)
    predictions, cursor = [], 0
    for batch_images, _ in loader:
        outputs = model([i.cuda() for i in batch_images])
        for out in outputs:
            image_id = images[cursor]["id"]
            cursor += 1
            for box, score, label in zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()):
                x0, y0, x1, y1 = box
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": float(score),
                    }
                )
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    data_dir = ROOT / cfg["data"]
    run_dir = ROOT / "runs" / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    images_val, _, categories = image_entries(data_dir, "val")
    nc = len(categories)
    model = build_model(cfg["model"], nc).cuda()

    t = cfg["training"]
    train_loader = DataLoader(
        DetectionDataset(data_dir, "train", build_augmentation(t["augment"])),
        batch_size=t["batch"],
        shuffle=True,
        num_workers=t["workers"],
        collate_fn=collate,
        persistent_workers=t["workers"] > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        DetectionDataset(data_dir, "val"), batch_size=1, num_workers=t["workers"], collate_fn=collate
    )

    run_info = {
        "name": cfg["name"],
        "framework": "torch",
        "model": cfg["model"],
        "nc": nc,
        "data": cfg["data"],
        "weights": str(run_dir / "best.pt"),
        "config": cfg,
    }
    history_path = run_dir / "history.csv"
    columns = ["epoch", "phase", "lr", "loss", "val_mAP50_95", "val_mAP50", "seconds", "peak_gpu_mb", "losses"]
    with history_path.open("w", newline="") as f:
        csv.writer(f).writerow(columns)

    accumulate = t["accumulate"]
    best_map, best_epoch, epoch_global = -1.0, None, 0
    start = time.perf_counter()
    trainability = []
    torch.cuda.reset_peak_memory_stats()

    for phase_index, phase in enumerate(cfg["phases"]):
        stale = 0
        if phase_index > 0:  # cada fase parte del mejor modelo hasta ahora
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cuda", weights_only=True))
        model.configure_trainable(phase)
        trainability.append({"phase": phase_index, "groups": model.trainability_report()})
        write_json(run_dir / "freeze_report.json", trainability)
        optimizer = build_optimizer(model, phase)
        steps_per_epoch = max(len(train_loader) // accumulate, 1)
        scheduler = build_scheduler(optimizer, phase, steps_per_epoch)

        for _ in range(phase["epochs"]):
            epoch_start = time.perf_counter()
            model.train()
            totals, n_batches = {}, 0
            optimizer.zero_grad(set_to_none=True)
            for step, (images, targets) in enumerate(train_loader, 1):
                images = [i.cuda() for i in images]
                targets = [{k: v.cuda() for k, v in tgt.items()} for tgt in targets]
                losses = model(images, targets)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"perdida no finita en epoca {epoch_global + 1}: {losses}")
                (loss / accumulate).backward()
                for k, v in losses.items():
                    totals[k] = totals.get(k, 0.0) + v.item()
                n_batches += 1
                if step % accumulate == 0:
                    if t["grad_clip"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            epoch_global += 1
            metrics = evaluate(data_dir / "annotations" / "val.json", predict_val(model, val_loader, images_val))
            score = metrics["mAP50_95"]
            improved = score > best_map
            if improved:
                best_map, best_epoch, stale = score, epoch_global, 0
                torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                stale += 1
            torch.save(model.state_dict(), run_dir / "last.pt")

            loss_summary = {k: round(v / max(n_batches, 1), 4) for k, v in totals.items()}
            peak = torch.cuda.max_memory_reserved() / 1024**2
            with history_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch_global,
                        phase_index,
                        optimizer.param_groups[0]["lr"],
                        round(sum(loss_summary.values()), 4),
                        round(score, 5),
                        round(metrics["mAP50"], 5),
                        round(time.perf_counter() - epoch_start, 1),
                        round(peak),
                        loss_summary,
                    ]
                )
            print(
                f"epoca {epoch_global} fase {phase_index}: perdida={sum(loss_summary.values()):.3f} "
                f"val mAP50-95={score:.4f} mAP50={metrics['mAP50']:.4f} "
                f"{time.perf_counter() - epoch_start:.0f}s GPU={peak:.0f}MB",
                flush=True,
            )
            write_json(
                run_dir / "run.json",
                {
                    **run_info,
                    "train_seconds": time.perf_counter() - start,
                    "best_epoch": best_epoch,
                    "epochs_run": epoch_global,
                    "peak_gpu_mb": peak,
                },
            )
            if t["patience"] and stale >= t["patience"]:  # termina la fase, no el entrenamiento
                print(f"detencion temprana de la fase {phase_index} en epoca {epoch_global}", flush=True)
                break


if __name__ == "__main__":
    main()
