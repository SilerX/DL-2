"""Comprueba el evaluador con casos de resultado conocido. Uso: python tests/test_eval.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection_eval import evaluate

BOXES = [(1, 0, [10, 10, 100, 100]), (1, 1, [300, 300, 50, 80]), (2, 0, [50, 60, 200, 150]), (3, 1, [400, 100, 60, 60])]


def write_gt(folder):
    gt = {
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": 640, "height": 480} for i in (1, 2, 3)],
        "annotations": [
            {"id": n + 1, "image_id": i, "category_id": c, "bbox": b, "area": b[2] * b[3], "iscrowd": 0}
            for n, (i, c, b) in enumerate(BOXES)
        ],
        "categories": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}, {"id": 2, "name": "sin_cajas"}],
    }
    path = Path(folder) / "gt.json"
    path.write_text(json.dumps(gt))
    return path


def det(i, c, b, s):
    return {"image_id": i, "category_id": c, "bbox": b, "score": s}


def close(a, b, tol=1e-3):
    assert abs(a - b) < tol, f"{a} != {b}"


with tempfile.TemporaryDirectory() as folder:
    gt = write_gt(folder)

    perfect = evaluate(gt, [det(i, c, b, 0.9) for i, c, b in BOXES])
    close(perfect["mAP50_95"], 1.0)
    close(perfect["mAP50"], 1.0)
    close(perfect["operating_point"]["precision"], 1.0)
    close(perfect["operating_point"]["recall"], 1.0)
    assert perfect["n_classes_evaluated"] == 2, "una clase sin cajas no debe promediarse"

    # Dos aciertos de cuatro cajas, mas un falso positivo con menor confianza
    partial = evaluate(
        gt,
        [
            det(1, 0, BOXES[0][2], 0.9),
            det(1, 1, BOXES[1][2], 0.8),
            det(3, 0, [0, 0, 50, 50], 0.3),
        ],
    )
    point = partial["operating_point"]
    close(point["precision"], 1.0)
    close(point["recall"], 0.5)
    close(point["conf"], 0.8)

    # Con umbral fijo bajo entra el falso positivo: precision 2/3
    fixed = evaluate(
        gt,
        [det(1, 0, BOXES[0][2], 0.9), det(1, 1, BOXES[1][2], 0.8), det(3, 0, [0, 0, 50, 50], 0.3)],
        conf=0.2,
    )
    close(fixed["operating_point"]["precision"], 2 / 3)

    # Clase equivocada: no cuenta como acierto y aparece en la confusion (real a, predicha b)
    wrong = evaluate(gt, [det(1, 1, BOXES[0][2], 0.9)])
    close(wrong["mAP50"], 0.0)
    assert wrong["confusion"]["matrix"][0][1] == 1

    # IoU bajo 0.5: no empareja (falso positivo y falso negativo)
    shifted = evaluate(gt, [det(1, 0, [80, 80, 100, 100], 0.9)])
    close(shifted["mAP50"], 0.0)

    assert evaluate(gt, [])["operating_point"]["conf"] is None

print("evaluador OK")
