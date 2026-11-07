"""Genera un dataset derivado (el original no se modifica) con train/val/test.

Salida en --out:
  images/<split>/*.jpg      imagenes reducidas (enlaces duros a --cache)
  labels/<split>/*.txt      labels formato YOLO
  annotations/<split>.json  mismas cajas en formato COCO (evaluador comun)
  splits/<split>.txt        ids de cada particion (se entregan con la tarea)
  data.yaml, manifest.json

Las decisiones (estrategia, fracciones, clases) entran por argumentos.
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

SPLITS = ("train", "val", "test")


def source_index(root):
    """Empareja imagen y label por id, sin importar en que carpeta esten.

    El original trae la imagen 2515 en train y su label en val.
    """
    images, labels = {}, {}
    origin = {}
    for p in sorted((root / "images").glob("*/*.jpg")):
        if p.stem in images:
            raise ValueError(f"id repetido entre particiones: {p.stem}")
        images[p.stem] = p
        origin[p.stem] = p.parent.name
    for p in sorted((root / "labels").glob("*/*.txt")):
        if p.stem in labels:
            raise ValueError(f"label repetido entre particiones: {p.stem}")
        labels[p.stem] = p
    moved = [s for s in images if s in labels and labels[s].parent.name != origin[s]]
    missing = sorted(set(images) ^ set(labels))
    if missing:
        raise ValueError(f"sin pareja imagen/label: {missing[:10]}")
    return images, labels, origin, moved


def clean_names(root):
    raw = yaml.safe_load((root / "data.yaml").read_text())["names"]
    return [re.sub(r"^\d+:\s*", "", n) for n in raw]


def load_class_map(path, names):
    """Identidad si no hay archivo. YAML: {names: [...], map: {clase_original: nueva o null}}."""
    if path is None:
        return names, {i: i for i in range(len(names))}
    spec = yaml.safe_load(Path(path).read_text())
    new_names = spec["names"]
    mapping = {names.index(k) if isinstance(k, str) else int(k): v for k, v in spec["map"].items()}
    for old in range(len(names)):
        if old not in mapping:
            raise ValueError(f"la clase {names[old]} no esta en el mapa")
        new = mapping[old]
        new = new_names.index(new) if isinstance(new, str) else new
        mapping[old] = new
    return new_names, mapping


def read_boxes(label_path, mapping):
    boxes = []
    for row in label_path.read_text().splitlines():
        if not row.strip():
            continue
        cls, x, y, w, h = row.split()
        new = mapping[int(float(cls))]
        if new is not None:
            boxes.append((new, float(x), float(y), float(w), float(h)))
    return boxes


def build_groups(stems, duplicates_json):
    """Cada imagen sin duplicados es un grupo de una sola imagen."""
    available = set(stems)
    groups, seen = [], set()
    if duplicates_json:
        for g in json.loads(Path(duplicates_json).read_text())["groups"]:
            members = [s for s in g if s in available]
            if members:
                groups.append(members)
                seen.update(members)
    groups.extend([s] for s in stems if s not in seen)
    return groups


def stratified_assign(groups, boxes_by_stem, fractions, seed):
    """Reparte grupos completos balanceando clases: parte por los que tienen la clase mas rara."""
    rng = random.Random(seed)
    rng.shuffle(groups)
    totals = Counter(b[0] for g in groups for s in g for b in boxes_by_stem[s])
    n_images = sum(map(len, groups))
    wanted_images = {k: f * n_images for k, f in fractions.items()}
    wanted_class = {k: {c: f * n for c, n in totals.items()} for k, f in fractions.items()}
    have_images = Counter()
    have_class = {k: Counter() for k in fractions}

    def group_classes(g):
        return Counter(b[0] for s in g for b in boxes_by_stem[s])

    def rarity(g):
        cs = group_classes(g)
        return (min((totals[c] for c in cs), default=10**9), -len(g))

    assignment = {}
    for g in sorted(groups, key=rarity):
        cs = group_classes(g)
        rare = min(cs, key=lambda c: totals[c]) if cs else None

        def score(k):
            fits = have_images[k] + len(g) <= wanted_images[k]
            need = wanted_class[k][rare] - have_class[k][rare] if rare is not None else 0
            return (fits, need, wanted_images[k] - have_images[k])

        target = max(fractions, key=score)
        have_images[target] += len(g)
        have_class[target].update(cs)
        for s in g:
            assignment[s] = target
    return assignment


def make_assignment(args, stems, origin, boxes_by_stem):
    if args.strategy == "regroup":
        fractions = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
        return stratified_assign(build_groups(stems, args.duplicates), boxes_by_stem, fractions, args.seed)
    # keep_val: train original intacto; el val original se divide en val y test
    assignment = {s: "train" for s in stems if origin[s] == "train"}
    val_stems = [s for s in stems if origin[s] == "val"]
    fractions = {"val": 1 - args.test_frac, "test": args.test_frac}
    assignment.update(stratified_assign(build_groups(val_stems, args.duplicates), boxes_by_stem, fractions, args.seed))
    return assignment


def cached_image(src, cache, max_side):
    """Reduce solo si excede max_side. Devuelve (ruta, ancho, alto)."""
    dst = cache / src.name
    if not dst.exists():
        with Image.open(src) as im:
            w, h = im.size
            if max(w, h) > max_side:
                im.draft("RGB", (max_side, max_side))
                scale = max_side / max(w, h)
                im = im.convert("RGB").resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            else:
                im = im.convert("RGB")
            tmp = dst.with_suffix(".tmp")
            im.save(tmp, "JPEG", quality=95)
            os.replace(tmp, dst)
    with Image.open(dst) as im:
        return dst, im.width, im.height


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def coco_records(stem, width, height, boxes, ann_id):
    anns = []
    for cls, x, y, w, h in boxes:
        x0, y0 = max(0.0, (x - w / 2) * width), max(0.0, (y - h / 2) * height)
        x1, y1 = min(float(width), (x + w / 2) * width), min(float(height), (y + h / 2) * height)
        anns.append(
            {
                "id": ann_id + len(anns),
                "image_id": int(stem),
                "category_id": cls,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
            }
        )
    return anns


def group_leakage(assignment, duplicates_json):
    if not duplicates_json:
        return None
    crossing = images = 0
    for g in json.loads(Path(duplicates_json).read_text())["groups"]:
        found = {assignment[s] for s in g if s in assignment}
        if len(found) > 1:
            crossing += 1
            images += sum(s in assignment for s in g)
    return {"groups_crossing_splits": crossing, "images_in_them": images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--strategy", choices=["keep_val", "regroup"], required=True)
    ap.add_argument("--train-frac", type=float)
    ap.add_argument("--val-frac", type=float)
    ap.add_argument("--test-frac", type=float, required=True)
    ap.add_argument("--duplicates", type=Path, help="JSON de check_duplicates.py; sus grupos no se separan")
    ap.add_argument("--class-map", type=Path)
    ap.add_argument("--max-side", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int, help="solo para pruebas de flujo")
    args = ap.parse_args()
    if args.strategy == "regroup" and None in (args.train_frac, args.val_frac):
        ap.error("regroup requiere --train-frac y --val-frac")

    images, labels, origin, moved = source_index(args.root)
    names = clean_names(args.root)
    new_names, mapping = load_class_map(args.class_map, names)

    stems = sorted(images, key=int)
    if args.limit:
        stems = random.Random(args.seed).sample(stems, args.limit)
    boxes_by_stem = {s: read_boxes(labels[s], mapping) for s in stems}
    assignment = make_assignment(args, stems, origin, boxes_by_stem)

    args.cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        cached = dict(zip(stems, pool.map(lambda s: cached_image(images[s], args.cache, args.max_side), stems)))

    if args.out.exists():
        shutil.rmtree(args.out)
    for sub in ("images", "labels"):
        for split in SPLITS:
            (args.out / sub / split).mkdir(parents=True)
    (args.out / "annotations").mkdir()
    (args.out / "splits").mkdir()

    summary = {}
    for split in SPLITS:
        members = sorted((s for s in stems if assignment[s] == split), key=int)
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in enumerate(new_names)],
        }
        counts = Counter()
        for s in members:
            path, width, height = cached[s]
            link_or_copy(path, args.out / "images" / split / path.name)
            boxes = boxes_by_stem[s]
            (args.out / "labels" / split / f"{s}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes)
            )
            coco["images"].append({"id": int(s), "file_name": path.name, "width": width, "height": height})
            coco["annotations"].extend(coco_records(s, width, height, boxes, len(coco["annotations"]) + 1))
            counts.update(b[0] for b in boxes)
        (args.out / "annotations" / f"{split}.json").write_text(json.dumps(coco))
        (args.out / "splits" / f"{split}.txt").write_text("\n".join(members) + "\n")
        summary[split] = {
            "images": len(members),
            "boxes": sum(counts.values()),
            "per_class": {new_names[c]: counts[c] for c in range(len(new_names))},
        }

    (args.out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(args.out.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": dict(enumerate(new_names)),
            },
            sort_keys=False,
        )
    )
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "labels_moved_to_match_image": moved,
        "leakage_by_groups": group_leakage(assignment, args.duplicates),
        "splits": summary,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    for split in SPLITS:
        print(split, summary[split]["images"], "imagenes,", summary[split]["boxes"], "cajas")
    print("fuga por grupos:", manifest["leakage_by_groups"])
    print("labels reubicados:", moved)


if __name__ == "__main__":
    main()
