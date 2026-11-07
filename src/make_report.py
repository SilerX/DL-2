"""Genera tablas y graficos a partir de runs/*/evaluation.json. No interpreta resultados.

Uso: make_report.py --runs runs/a runs/b --labels "CNN propia" "YOLO" --out results
El color de cada modelo depende de su posicion en --runs, asi que mantener el orden entre figuras.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "savefig.dpi": 200,
    }
)

COLUMNS = [
    ("Modelo", lambda e: e["label"]),
    ("mAP@0.5:0.95", lambda e: f"{e['test']['mAP50_95']:.3f}"),
    ("mAP@0.5", lambda e: f"{e['test']['mAP50']:.3f}"),
    ("mAP@0.75", lambda e: f"{e['test']['mAP75']:.3f}"),
    ("mAP pequeños", lambda e: f"{e['test']['mAP_small']:.3f}"),
    ("mAP medianos", lambda e: f"{e['test']['mAP_medium']:.3f}"),
    ("mAP grandes", lambda e: f"{e['test']['mAP_large']:.3f}"),
    ("AR@100", lambda e: f"{e['test']['AR100']:.3f}"),
    ("Precisión", lambda e: f"{e['test']['operating_point']['precision']:.3f}"),
    ("Recall", lambda e: f"{e['test']['operating_point']['recall']:.3f}"),
    ("F1", lambda e: f"{e['test']['operating_point']['f1']:.3f}"),
    ("Umbral (de val)", lambda e: f"{e['conf_from_val']:.3f}"),
    ("Entrenamiento (min)", lambda e: f"{e['train_seconds'] / 60:.1f}"),
    ("Épocas", lambda e: e["epochs_run"]),
    ("Mejor época", lambda e: e["best_epoch"]),
    ("Latencia (ms/img)", lambda e: f"{e['latency_ms']:.1f}"),
    ("FPS", lambda e: f"{e['fps']:.1f}"),
    ("Parámetros (M)", lambda e: f"{e['num_parameters'] / 1e6:.2f}"),
    ("Pesos (MB)", lambda e: f"{e['weights_mb']:.1f}"),
    ("GPU pico (MB)", lambda e: "-" if e["peak_gpu_mb"] is None else f"{e['peak_gpu_mb']:.0f}"),
]


def load_runs(runs, labels):
    entries = []
    for run, label in zip(runs, labels):
        entry = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
        entry["label"] = label
        entry["dir"] = run
        entries.append(entry)
    return entries


def write_tables(entries, out):
    header = [c[0] for c in COLUMNS]
    rows = [[str(fn(e)) for _, fn in COLUMNS] for e in entries]
    with (out / "comparacion.csv").open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([header, *rows])
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out / "comparacion.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    classes = sorted({c for e in entries for c in e["test"]["per_class"]})
    with (out / "ap_por_clase.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clase", "cajas_test", *[f"AP50 {e['label']}" for e in entries]])
        for c in classes:
            n = next(e["test"]["per_class"][c]["n_gt"] for e in entries if c in e["test"]["per_class"])
            w.writerow([c, n, *[f"{e['test']['per_class'].get(c, {}).get('AP50', 0):.3f}" for e in entries]])


def bars(ax, labels, values, title, fmt, x_max=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.55, color=SERIES[0])
    ax.set_yticks(y, labels, color=INK2)
    ax.set_title(title, color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    top = x_max or (max(values) if max(values) > 0 else 1)
    ax.set_xlim(0, top * 1.18)
    for yi, v in zip(y, values):
        ax.text(v + top * 0.015, yi, fmt.format(v), va="center", color=INK, fontsize=10)


def fig_metrics(entries, out):
    labels = [e["label"] for e in entries]
    panels = [
        ("mAP@0.5:0.95", [e["test"]["mAP50_95"] for e in entries], "{:.3f}"),
        ("mAP@0.5", [e["test"]["mAP50"] for e in entries], "{:.3f}"),
        ("Precisión", [e["test"]["operating_point"]["precision"] for e in entries], "{:.3f}"),
        ("Recall", [e["test"]["operating_point"]["recall"] for e in entries], "{:.3f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 1.1 + 1.0 * len(entries) * 2), sharey=True)
    for ax, (title, values, fmt) in zip(axes.flat, panels):
        bars(ax, labels, values, title, fmt, x_max=1)  # misma escala 0-1 en todos los paneles
    fig.tight_layout()
    fig.savefig(out / "metricas_test.png")
    plt.close(fig)


def fig_costs(entries, out):
    labels = [e["label"] for e in entries]
    panels = [
        ("Tiempo de entrenamiento (min)", [e["train_seconds"] / 60 for e in entries], "{:.1f}"),
        ("Latencia (ms por imagen)", [e["latency_ms"] for e in entries], "{:.1f}"),
        ("Parámetros (millones)", [e["num_parameters"] / 1e6 for e in entries], "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.2 + 0.8 * len(entries)), sharey=True)
    for ax, (title, values, fmt) in zip(axes, panels):
        bars(ax, labels, values, title, fmt)
    fig.tight_layout()
    fig.savefig(out / "costos.png")
    plt.close(fig)


def fig_size_breakdown(entries, out):
    labels = [e["label"] for e in entries]
    panels = [
        ("mAP objetos pequeños", "mAP_small"),
        ("mAP objetos medianos", "mAP_medium"),
        ("mAP objetos grandes", "mAP_large"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.2 + 0.8 * len(entries)), sharey=True)
    for ax, (title, key) in zip(axes, panels):
        bars(ax, labels, [max(e["test"][key], 0) for e in entries], title, "{:.3f}", x_max=1)
    fig.tight_layout()
    fig.savefig(out / "map_por_tamano.png")
    plt.close(fig)


def fig_tradeoff(entries, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, e in enumerate(entries):
        ax.scatter(
            e["latency_ms"], e["test"]["mAP50_95"], s=90, color=SERIES[i], edgecolors=SURFACE, linewidths=2, zorder=3
        )
        ax.annotate(
            e["label"], (e["latency_ms"], e["test"]["mAP50_95"]), xytext=(8, 6), textcoords="offset points", color=INK2
        )
    ax.set_xlabel("Latencia (ms por imagen)")
    ax.set_ylabel("mAP@0.5:0.95 en test")
    ax.set_title("Exactitud y velocidad por modelo")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.3, y=0.2)
    fig.tight_layout()
    fig.savefig(out / "exactitud_vs_latencia.png")
    plt.close(fig)


def learning_curve(entry):
    """Devuelve (epocas, mAP50-95 en val) con el mAP interno de cada framework durante el entrenamiento."""
    run = entry["dir"]
    if (run / "results.csv").exists():
        with (run / "results.csv").open() as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        return [int(r["epoch"]) for r in rows], [float(r["metrics/mAP50-95(B)"]) for r in rows]
    with (run / "history.csv").open() as f:
        rows = list(csv.DictReader(f))
    return [int(r["epoch"]) for r in rows], [float(r["val_mAP50_95"]) for r in rows]


def fig_curves(entries, out):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, e in enumerate(entries):
        epochs, values = learning_curve(e)
        ax.plot(epochs, values, color=SERIES[i], lw=2, label=e["label"], marker="o" if len(epochs) < 3 else None, ms=8)
    ax.set_xlabel("Época")
    ax.set_ylabel("mAP@0.5:0.95 en val")
    ax.set_title("Curvas de aprendizaje")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "curvas_aprendizaje.png")
    plt.close(fig)


def fig_pr(entries, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, e in enumerate(entries):
        curve = e["test"].get("pr_curve")
        if curve:
            ax.plot(curve["recall"], curve["precision"], color=SERIES[i], lw=2, label=e["label"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Curva precisión-recall micro en test (IoU 0.5)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "curva_pr.png")
    plt.close(fig)


def fig_per_class(entries, out):
    classes = sorted(
        {c for e in entries for c in e["test"]["per_class"]},
        key=lambda c: -next(e["test"]["per_class"][c]["n_gt"] for e in entries if c in e["test"]["per_class"]),
    )
    if not classes:
        return
    matrix = np.array([[e["test"]["per_class"].get(c, {}).get("AP50", np.nan) for e in entries] for c in classes])
    counts = [next(e["test"]["per_class"][c]["n_gt"] for e in entries if c in e["test"]["per_class"]) for c in classes]
    fig, ax = plt.subplots(figsize=(3 + 1.6 * len(entries), 1.6 + 0.34 * len(classes)))
    image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(entries)), [e["label"] for e in entries], color=INK2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(classes)), [f"{c} ({n})" for c, n in zip(classes, counts)], color=INK2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for (r, c), v in np.ndenumerate(matrix):
        if not np.isnan(v):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#ffffff" if v > 0.55 else INK)
    ax.set_title("AP@0.5 por clase en test (cajas reales entre paréntesis)", pad=32)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "ap_por_clase.png")
    plt.close(fig)


def fig_confusion(entries, out):
    for e in entries:
        conf = e["test"].get("confusion")
        if not conf:
            continue
        matrix = np.array(conf["matrix"])
        keep = [i for i in range(len(matrix)) if matrix[i].sum() + matrix[:, i].sum() > 0]
        matrix = matrix[np.ix_(keep, keep)]
        labels = [conf["labels"][i] for i in keep]
        size = 3 + 0.42 * len(keep)
        fig, ax = plt.subplots(figsize=(size + 1, size))
        ax.imshow(matrix, cmap=SEQUENTIAL, norm=PowerNorm(0.4, vmin=0, vmax=max(matrix.max(), 1)))
        ax.set_xticks(range(len(keep)), labels, rotation=90, color=INK2, fontsize=9)
        ax.set_yticks(range(len(keep)), labels, color=INK2, fontsize=9)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        limit = matrix.max() * 0.3
        for (r, c), v in np.ndenumerate(matrix):
            if v:
                ax.text(c, r, str(v), ha="center", va="center", fontsize=8, color="#ffffff" if v > limit else INK)
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusión en test: {e['label']}")
        fig.tight_layout()
        safe = "".join(ch if ch.isalnum() else "_" for ch in e["label"])
        fig.savefig(out / f"confusion_{safe}.png")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.runs) != len(args.labels):
        ap.error("--runs y --labels deben tener el mismo largo")
    args.out.mkdir(parents=True, exist_ok=True)

    entries = load_runs(args.runs, args.labels)
    write_tables(entries, args.out)
    fig_metrics(entries, args.out)
    fig_costs(entries, args.out)
    fig_size_breakdown(entries, args.out)
    fig_tradeoff(entries, args.out)
    fig_curves(entries, args.out)
    fig_pr(entries, args.out)
    fig_per_class(entries, args.out)
    fig_confusion(entries, args.out)
    print((args.out / "comparacion.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
