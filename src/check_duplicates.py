"""Busca imagenes casi identicas (hash perceptual) dentro y entre particiones.

Guarda los grupos en un JSON que prepare_dataset.py usa para no separar un
mismo grupo entre particiones.
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw


def phash(path):
    with Image.open(path) as im:
        im.draft("L", (64, 64))
        return path, imagehash.phash(im.convert("L"), hash_size=8).hash.flatten()


def find_groups(bits, threshold, extra_pairs=()):
    """Union-find sobre pares con distancia de Hamming <= threshold, mas extra_pairs."""
    parent = list(range(len(bits)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    distances = Counter()
    for i in range(len(bits) - 1):
        d = (bits[i + 1 :] != bits[i]).sum(axis=1)
        distances.update(d.tolist())
        for j in np.flatnonzero(d <= threshold) + i + 1:
            parent[root(i)] = root(int(j))
    for i, j in extra_pairs:
        parent[root(i)] = root(j)
    groups = {}
    for i in range(len(bits)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values()), distances


def contact_sheet(pairs, out, thumb=(320, 180)):
    pairs = pairs[:12]
    sheet = Image.new("RGB", (thumb[0] * 2, (thumb[1] + 18) * len(pairs)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (a, b) in enumerate(pairs):
        y = row * (thumb[1] + 18)
        for col, p in enumerate((a, b)):
            with Image.open(p) as im:
                im.draft("RGB", thumb)
                sheet.paste(im.convert("RGB").resize(thumb), (col * thumb[0], y + 18))
            draw.text((col * thumb[0] + 4, y + 3), f"{p.parent.name}/{p.name}", fill="black")
    sheet.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument(
        "--consecutive-limit",
        type=int,
        help="une tambien ids consecutivos con distancia <= este limite (frames de un mismo clip)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted((args.root / "images").glob("*/*.jpg"))
    with ThreadPoolExecutor(8) as pool:
        hashed = list(pool.map(phash, paths))
    paths = [p for p, _ in hashed]
    bits = np.stack([h for _, h in hashed])

    extra_pairs = []
    if args.consecutive_limit is not None:
        by_id = {int(p.stem): i for i, p in enumerate(paths)}
        for image_id, i in by_id.items():
            j = by_id.get(image_id + 1)
            if j is not None and (bits[i] != bits[j]).sum() <= args.consecutive_limit:
                extra_pairs.append((i, j))
        print(f"ids consecutivos unidos: {len(extra_pairs)} | limite: {args.consecutive_limit}")

    groups, distances = find_groups(bits, args.threshold, extra_pairs)
    multi = [g for g in groups if len(g) > 1]
    cross = [g for g in multi if len({paths[i].parent.name for i in g}) > 1]

    print(f"imagenes: {len(paths)} | umbral Hamming: {args.threshold}")
    print(f"grupos con >1 imagen: {len(multi)} | imagenes en ellos: {sum(map(len, multi))}")
    print(f"grupos que cruzan train/val: {len(cross)} | imagenes en ellos: {sum(map(len, cross))}")
    print("tamanio de grupo (top):", sorted(map(len, multi), reverse=True)[:10])
    print("pares por distancia 0..12:", [distances[d] for d in range(13)])

    (args.out / "duplicate_groups.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "consecutive_limit": args.consecutive_limit,
                "groups": [sorted(paths[i].stem for i in g) for g in multi],
            },
            indent=1,
        )
    )
    pairs = []
    for g in cross:
        by_split = {}
        for i in g:
            by_split.setdefault(paths[i].parent.name, paths[i])
        pairs.append(tuple(by_split.values())[:2])
    if pairs:
        contact_sheet(pairs, args.out / "duplicate_pairs.jpg")


if __name__ == "__main__":
    main()
