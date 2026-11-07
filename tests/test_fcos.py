"""Comprueba la asignacion y las perdidas de FCOS en CPU. Uso: python tests/test_fcos.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from custom_cnn import CustomDetector, ResNetBackbone, norm_layer, assign_targets, centerness_target, iou_loss

NC = 3
POINTS = torch.tensor([[50.0, 50.0], [20.0, 20.0], [150.0, 150.0], [55.0, 45.0]])

# Punto dentro de la caja: etiqueta correcta y distancias ltrb exactas; punto fuera: fondo
boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
labels, ltrb = assign_targets(POINTS, boxes, torch.tensor([2]), 8, (0, 1000), 0, NC)
assert labels.tolist() == [2, 2, NC, 2]
assert ltrb[0].tolist() == [50, 50, 50, 50]
assert ltrb[3].tolist() == [55, 45, 45, 55]

# Con radio central solo quedan los puntos cerca del centro
labels, _ = assign_targets(POINTS, boxes, torch.tensor([2]), 8, (0, 1000), 1.0, NC)
assert labels.tolist() == [2, NC, NC, 2]

# El rango de tamanio del nivel excluye cajas fuera de escala
labels, _ = assign_targets(POINTS, boxes, torch.tensor([2]), 8, (0, 10), 0, NC)
assert (labels == NC).all()

# Dos cajas sobre el mismo punto: gana la de menor area
boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [40.0, 40.0, 60.0, 60.0]])
labels, _ = assign_targets(POINTS[:1], boxes, torch.tensor([0, 1]), 8, (0, 1000), 0, NC)
assert labels.tolist() == [1]

# Sin cajas todo es fondo
labels, _ = assign_targets(POINTS, torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64), 8, (0, 1000), 0, NC)
assert (labels == NC).all()

assert abs(centerness_target(torch.tensor([[50.0, 50.0, 50.0, 50.0]])).item() - 1.0) < 1e-6

model = CustomDetector(
    nc=NC,
    stage_channels=[8, 16, 16, 32, 32],
    stage_blocks=[1, 1, 1, 1, 1],
    block="residual",
    norm="group",
    fpn_channels=16,
    fpn_from_stage=2,
    fpn_extra_levels=0,
    head_convs=1,
    size_ranges=[[0, 64], [64, 128], [128, 100000]],
    center_radius=1.5,
    use_centerness=True,
    focal_alpha=0.25,
    focal_gamma=2.0,
    cls_weight=1.0,
    box_weight=1.0,
    min_size=160,
    max_size=160,
    nms_iou=0.5,
    smooth_l1_beta=0.1,
)
assert model.strides == [8, 16, 32]
images = [torch.rand(3, 120, 160), torch.rand(3, 90, 160)]
targets = [
    {"boxes": torch.tensor([[10.0, 10.0, 60.0, 80.0]]), "labels": torch.tensor([1])},
    {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)},
]

model.train()
losses = model(images, targets)
assert set(losses) == {"cls", "box", "centerness"}
assert all(torch.isfinite(v) for v in losses.values())
sum(losses.values()).backward()
assert all(p.grad is not None for p in model.parameters())

# Un lote sin ninguna caja no debe romper las perdidas
empty = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}]
assert all(torch.isfinite(v) for v in model([images[0]], empty).values())

model.eval()
model.set_score_threshold(0.001)
with torch.inference_mode():
    outputs = model(images)
assert len(outputs) == 2
for image, output in zip(images, outputs):
    assert output["boxes"].shape[1] == 4
    assert len(output["boxes"]) <= 100
    assert (output["labels"] < NC).all()
    if len(output["boxes"]):
        assert output["boxes"][:, 2].max() <= image.shape[2] + 1e-3
        assert output["boxes"][:, 3].max() <= image.shape[1] + 1e-3

# Perdidas IoU: cero si la prediccion es exacta, crecen si se aleja
same = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
for kind in ("iou", "giou"):
    assert iou_loss(same, same, kind).item() < 1e-5
    assert iou_loss(same * 2, same, kind).item() > iou_loss(same * 1.1, same, kind).item() > 0
# Cajas 2x2 y 4x4 con el mismo punto en la esquina comun: IoU = 4/16
small, big = torch.tensor([[0.0, 0.0, 2.0, 2.0]]), torch.tensor([[0.0, 0.0, 4.0, 4.0]])
assert abs(iou_loss(small, big, "iou").item() + torch.log(torch.tensor(0.25)).item()) < 1e-5

# ResNet-50: 23,5 M parametros en el backbone (sin la capa de clasificacion de ImageNet) y strides 4-32
resnet50 = ResNetBackbone([256, 512, 1024, 2048], [3, 4, 6, 3], norm_layer("group", 32))
assert resnet50.strides == [4, 8, 16, 32]
assert sum(p.numel() for p in resnet50.parameters()) == 23_508_032
assert [f.shape[1:] for f in resnet50(torch.rand(1, 3, 64, 64))] == [
    (256, 16, 16), (512, 8, 8), (1024, 4, 4), (2048, 2, 2)
]


def overfit(model):
    """Sobreajuste a una imagen: si perdida y asignacion estan bien, debe poder encontrar la caja."""
    torch.manual_seed(0)
    image = torch.rand(3, 160, 160)
    box = torch.tensor([[40.0, 40.0, 120.0, 120.0]])
    image[:, 40:120, 40:120] = 0.9
    target = [{"boxes": box, "labels": torch.tensor([1])}]
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(250):
        loss = sum(model([image], target).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    model.set_score_threshold(0.001)
    with torch.inference_mode():
        output = model([image])[0]
    assert len(output["boxes"]), "sin detecciones tras sobreajustar"
    top = output["scores"].argmax()
    assert output["labels"][top].item() == 1
    ix = (torch.min(output["boxes"][top, 2:], box[0, 2:]) - torch.max(output["boxes"][top, :2], box[0, :2])).clamp(min=0).prod()
    area_a = (output["boxes"][top, 2:] - output["boxes"][top, :2]).prod()
    area_b = (box[0, 2:] - box[0, :2]).prod()
    iou = (ix / (area_a + area_b - ix)).item()
    assert iou > 0.5, f"IoU tras sobreajustar: {iou:.2f}"
    return iou


print(f"FCOS OK (IoU tras sobreajustar a una imagen: {overfit(model):.2f})")

# Backbone bottleneck pequenio con perdida IoU y niveles extra (como P3-P7)
torch.manual_seed(0)
small_resnet = CustomDetector(
    nc=NC,
    stage_channels=[32, 64, 64, 64],
    stage_blocks=[1, 1, 1, 1],
    block="bottleneck",
    norm="group",
    fpn_channels=16,
    fpn_from_stage=1,
    fpn_extra_levels=2,
    head_convs=1,
    size_ranges=[[0, 32], [32, 64], [64, 128], [128, 256], [256, 100000]],
    center_radius=1.5,
    use_centerness=True,
    focal_alpha=0.25,
    focal_gamma=2.0,
    cls_weight=1.0,
    box_weight=1.0,
    min_size=160,
    max_size=160,
    nms_iou=0.5,
    norm_groups=4,
    box_loss="giou",
)
assert small_resnet.strides == [8, 16, 32, 64, 128]
print(f"FCOS bottleneck + GIoU OK (IoU tras sobreajustar a una imagen: {overfit(small_resnet):.2f})")
