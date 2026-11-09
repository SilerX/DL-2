"""Repite la latencia de una ejecucion ya evaluada, con la GPU sin otros procesos.

Mismo protocolo que evaluate_run.py, repetido --repeats veces en el mismo proceso.
latency_ms queda como la mediana; las repeticiones van en latency_ms_runs y la
medicion de evaluate_run.py en latency_ms_previous.
Uso: remeasure_latency.py --run runs/<nombre> --data data/<dataset>
"""
import argparse
import json
import statistics
from pathlib import Path

from common import write_json
from evaluate_run import measure_latency
from predictors import load_predictor, load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    path = args.run / "evaluation.json"
    ev = json.loads(path.read_text(encoding="utf-8"))
    predictor = load_predictor(load_run(args.run))
    runs = [measure_latency(predictor, args.data, "test", ev["conf_from_val"]) for _ in range(args.repeats)]
    latency = statistics.median(runs)
    ev.setdefault("latency_ms_previous", ev["latency_ms"])
    ev["latency_ms_runs"] = runs
    ev["latency_ms"] = latency
    ev["fps"] = 1000 / latency
    write_json(path, ev)
    print(f"{ev['name']}: latencia={latency:.1f}ms repeticiones={[round(r, 1) for r in runs]} (anterior {ev['latency_ms_previous']:.1f}ms)")


if __name__ == "__main__":
    main()
