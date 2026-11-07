"""Comprueba las particiones: reparto, clases y parecido entre imagenes de particiones distintas.

Para cada imagen de val y test busca la mas parecida en otra particion. El hash se calcula sobre las
imagenes originales, igual que check_duplicates.py, para que las distancias sean comparables con el
umbral de agrupacion. Los ids consecutivos suelen ser frames seguidos, asi que tambien se cuentan.
Uso: verify_split.py --data data/<nombre> --root <FjordFish> --out artifacts/<carpeta>
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from check_duplicates import contact_sheet, phash

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.data / "manifest.json").read_text())
    total = sum(manifest["splits"][s]["images"] for s in SPLITS)
    print("particion | imagenes | % | cajas")
    for s in SPLITS:
        info = manifest["splits"][s]
        print(f"{s} | {info['images']} | {100 * info['images'] / total:.1f} | {info['boxes']}")
    print("fuga por grupos:", manifest["leakage_by_groups"])

    print("\nclase | train | val | test")
    for name in manifest["splits"]["train"]["per_class"]:
        print(name, "|", " | ".join(str(manifest["splits"][s]["per_class"][name]) for s in SPLITS))

    originals = {p.stem: p for p in (args.root / "images").glob("*/*.jpg")}
    assignment = {}
    for s in SPLITS:
        for stem in (args.data / "splits" / f"{s}.txt").read_text().split():
            assignment[stem] = s
    stems = sorted(assignment, key=int)
    with ThreadPoolExecutor(8) as pool:
        bits = np.stack([h for _, h in pool.map(phash, [originals[s] for s in stems])])
    labels = np.array([assignment[s] for s in stems])

    def split_path(stem):
        return args.data / "images" / assignment[stem] / f"{stem}.jpg"

    results = {}
    for s in ("val", "test"):
        others = np.flatnonzero(labels != s)
        nearest = []
        for i in np.flatnonzero(labels == s):
            d = (bits[others] != bits[i]).sum(axis=1)
            j = int(d.argmin())
            nearest.append((int(d[j]), stems[i], stems[others[j]]))
        histogram = Counter(n[0] for n in nearest)
        results[s] = {"total": len(nearest), "histograma_distancia": dict(sorted(histogram.items()))}
        nearest.sort()
        contact_sheet(
            [(split_path(a), split_path(b)) for _, a, b in nearest[:12]], args.out / f"pares_mas_parecidos_{s}.jpg"
        )

    ids = {int(s) for s in stems}
    adjacent = [(i, i + 1) for i in sorted(ids) if i + 1 in ids]
    crossing = [(a, b) for a, b in adjacent if assignment[str(a)] != assignment[str(b)]]
    results["ids_consecutivos"] = {"pares": len(adjacent), "en_particiones_distintas": len(crossing)}
    if crossing:
        contact_sheet(
            [(split_path(str(a)), split_path(str(b))) for a, b in crossing[:12]],
            args.out / "ids_consecutivos_separados.jpg",
        )

    print("\ndistancia a la imagen mas parecida en otra particion:")
    print(json.dumps(results, indent=1))
    (args.out / "verificacion.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
