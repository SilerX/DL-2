"""CNN propia: detector FCOS (anchor-free) entrenado desde cero.

Diseno del grupo. Todas las medidas (backbone, canales, niveles, rangos, perdidas, pesos) entran
desde la configuracion. Interfaz definida en torch_models.py.

Backbones:
- block plain | residual: cada etapa reduce a la mitad con una convolucion de paso 2 (strides 2, 4, 8, ...).
- block bottleneck: ResNet (tallo 7x7 + max pool, bloques bottleneck de expansion 4, strides 4, 8, 16, 32);
  con stage_blocks [3, 4, 6, 3] y stage_channels [256, 512, 1024, 2048] es ResNet-50. Inicializacion He.
Perdida de caja: smooth_l1 sobre ltrb, iou (-ln IoU, FCOS original) o giou (1 - GIoU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import batched_nms, sigmoid_focal_loss

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PRIOR_PROBABILITY = 0.01  # inicializacion de la clasificacion de FCOS/RetinaNet
TOPK_PER_LEVEL = 1000


def norm_layer(kind, groups=8):
    """Devuelve una funcion canales -> capa de normalizacion."""
    if kind == "batch":
        return nn.BatchNorm2d
    if kind == "group":
        return lambda channels: nn.GroupNorm(min(groups, channels), channels)
    raise ValueError(f"norm desconocida: {kind}")


def conv_block(in_channels, out_channels, norm, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        norm(out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm):
        super().__init__()
        self.body = nn.Sequential(
            conv_block(channels, channels, norm),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            norm(channels),
        )

    def forward(self, x):
        return F.relu(x + self.body(x), inplace=True)


class Bottleneck(nn.Module):
    """Bloque de ResNet-50: 1x1 reduce, 3x3 (con el paso), 1x1 expande; atajo con proyeccion si cambia la forma."""

    def __init__(self, in_channels, out_channels, stride, norm):
        super().__init__()
        width = out_channels // 4
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 1, bias=False),
            norm(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride, 1, bias=False),
            norm(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, 1, bias=False),
            norm(out_channels),
        )
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), norm(out_channels)
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        return F.relu(identity + self.body(x), inplace=True)


class ResNetBackbone(nn.Module):
    STEM_CHANNELS = 64

    def __init__(self, stage_channels, stage_blocks, norm):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.STEM_CHANNELS, 7, 2, 3, bias=False),
            norm(self.STEM_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.stages = nn.ModuleList()
        previous = self.STEM_CHANNELS
        for i, (channels, blocks) in enumerate(zip(stage_channels, stage_blocks)):
            stride = 1 if i == 0 else 2
            layers = [Bottleneck(previous, channels, stride, norm)]
            layers += [Bottleneck(channels, channels, 1, norm) for _ in range(blocks - 1)]
            self.stages.append(nn.Sequential(*layers))
            previous = channels
        self.strides = [2 ** (i + 2) for i in range(len(stage_channels))]
        for m in self.modules():  # inicializacion He
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class Backbone(nn.Module):
    """Cada etapa reduce a la mitad con una convolucion de paso 2 y luego aplica sus bloques."""

    def __init__(self, stage_channels, stage_blocks, block, norm):
        super().__init__()
        self.strides = [2 ** (i + 1) for i in range(len(stage_channels))]
        self.stages = nn.ModuleList()
        previous = 3
        for channels, blocks in zip(stage_channels, stage_blocks):
            layers = [conv_block(previous, channels, norm, stride=2)]
            for _ in range(blocks):
                layers.append(ResidualBlock(channels, norm) if block == "residual" else conv_block(channels, channels, norm))
            self.stages.append(nn.Sequential(*layers))
            previous = channels

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class FPN(nn.Module):
    """Combina de arriba hacia abajo; n_out_levels agrega niveles extra por encima del ultimo."""

    def __init__(self, in_channels, out_channels, n_extra, norm):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
        self.smooth = nn.ModuleList(conv_block(out_channels, out_channels, norm) for _ in in_channels)
        self.extra = nn.ModuleList(conv_block(out_channels, out_channels, norm, stride=2) for _ in range(n_extra))

    def forward(self, features):
        maps = [lat(f) for lat, f in zip(self.lateral, features)]
        for i in range(len(maps) - 2, -1, -1):
            maps[i] = maps[i] + F.interpolate(maps[i + 1], size=maps[i].shape[-2:], mode="nearest")
        outputs = [s(m) for s, m in zip(self.smooth, maps)]
        for layer in self.extra:
            outputs.append(layer(outputs[-1]))
        return outputs


class Scale(nn.Module):
    """Factor aprendible por nivel para la regresion."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class Head(nn.Module):
    """Cabeza compartida entre niveles: clasificacion, regresion ltrb y centerness opcional."""

    def __init__(self, channels, nc, n_convs, n_levels, norm, use_centerness):
        super().__init__()
        self.cls_tower = nn.Sequential(*[conv_block(channels, channels, norm) for _ in range(n_convs)])
        self.reg_tower = nn.Sequential(*[conv_block(channels, channels, norm) for _ in range(n_convs)])
        self.cls_out = nn.Conv2d(channels, nc, 3, padding=1)
        self.reg_out = nn.Conv2d(channels, 4, 3, padding=1)
        self.ctr_out = nn.Conv2d(channels, 1, 3, padding=1) if use_centerness else None
        self.scales = nn.ModuleList(Scale() for _ in range(n_levels))
        prior = -torch.log(torch.tensor((1 - PRIOR_PROBABILITY) / PRIOR_PROBABILITY))
        nn.init.constant_(self.cls_out.bias, prior.item())

    def forward(self, features):
        cls_all, reg_all, ctr_all = [], [], []
        for feature, scale in zip(features, self.scales):
            cls_feature = self.cls_tower(feature)
            reg_feature = self.reg_tower(feature)
            cls_all.append(self.cls_out(cls_feature))
            # exp mantiene positivas las distancias; estan expresadas en unidades del stride del nivel
            reg_all.append(torch.exp(scale(self.reg_out(reg_feature)).clamp(max=10)))
            ctr_all.append(self.ctr_out(reg_feature) if self.ctr_out is not None else None)
        return cls_all, reg_all, ctr_all


def grid_points(feature, stride):
    h, w = feature.shape[-2:]
    ys = (torch.arange(h, device=feature.device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(w, device=feature.device, dtype=torch.float32) + 0.5) * stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.ravel(), yy.ravel()], dim=1)


def assign_targets(points, boxes, labels, stride, size_range, center_radius, nc):
    """Asignacion de FCOS en un nivel.

    Un punto es positivo si cae dentro de la caja (o de su region central si center_radius > 0)
    y su mayor distancia ltrb esta en size_range. Si hay varias cajas, gana la de menor area.
    Devuelve etiquetas (nc = fondo) y ltrb en pixeles.
    """
    n = len(points)
    target_labels = torch.full((n,), nc, dtype=torch.int64, device=points.device)
    target_ltrb = torch.zeros((n, 4), device=points.device)
    if len(boxes) == 0:
        return target_labels, target_ltrb

    x, y = points[:, 0, None], points[:, 1, None]
    x0, y0, x1, y1 = boxes.unbind(dim=1)
    ltrb = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
    if center_radius > 0:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = center_radius * stride
        region = torch.stack(
            [x - torch.max(x0, cx - r), y - torch.max(y0, cy - r), torch.min(x1, cx + r) - x, torch.min(y1, cy + r) - y],
            dim=2,
        )
        inside = region.min(dim=2).values > 0
    else:
        inside = ltrb.min(dim=2).values > 0
    longest = ltrb.max(dim=2).values
    in_range = (longest >= size_range[0]) & (longest <= size_range[1])
    areas = ((x1 - x0) * (y1 - y0)).expand(n, -1).clone()
    areas[~(inside & in_range)] = float("inf")
    best_area, best = areas.min(dim=1)
    positive = torch.isfinite(best_area)
    target_labels[positive] = labels[best[positive]]
    target_ltrb[positive] = ltrb[positive, best[positive]]
    return target_labels, target_ltrb


def iou_loss(pred, target, kind):
    """pred y target en ltrb respecto del mismo punto; la IoU no depende de la unidad."""
    pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
    w = torch.min(pred[:, 0], target[:, 0]) + torch.min(pred[:, 2], target[:, 2])
    h = torch.min(pred[:, 1], target[:, 1]) + torch.min(pred[:, 3], target[:, 3])
    intersection = w * h
    union = pred_area + target_area - intersection
    iou = (intersection + 1.0e-7) / (union + 1.0e-7)
    if kind == "iou":
        return -torch.log(iou.clamp(min=1.0e-7))
    enclosing = (torch.max(pred[:, 0], target[:, 0]) + torch.max(pred[:, 2], target[:, 2])) * (
        torch.max(pred[:, 1], target[:, 1]) + torch.max(pred[:, 3], target[:, 3])
    )
    return 1 - (iou - (enclosing - union) / enclosing.clamp(min=1.0e-7))


def centerness_target(ltrb):
    lr, tb = ltrb[:, [0, 2]], ltrb[:, [1, 3]]
    ratio = (lr.min(dim=1).values / lr.max(dim=1).values) * (tb.min(dim=1).values / tb.max(dim=1).values)
    return torch.sqrt(ratio.clamp(min=0))


class CustomDetector(nn.Module):
    def __init__(
        self,
        nc,
        stage_channels,
        stage_blocks,
        block,
        norm,
        fpn_channels,
        fpn_from_stage,
        fpn_extra_levels,
        head_convs,
        size_ranges,
        center_radius,
        use_centerness,
        focal_alpha,
        focal_gamma,
        cls_weight,
        box_weight,
        min_size,
        max_size,
        nms_iou,
        norm_groups=8,
        box_loss="smooth_l1",
        smooth_l1_beta=None,
    ):
        super().__init__()
        self.nc = nc
        self.fpn_from_stage = fpn_from_stage
        selected = stage_channels[fpn_from_stage:]
        n_levels = len(selected) + fpn_extra_levels
        norm = norm_layer(norm, norm_groups)
        if block == "bottleneck":
            self.backbone = ResNetBackbone(stage_channels, stage_blocks, norm)
        else:
            self.backbone = Backbone(stage_channels, stage_blocks, block, norm)
        first = self.backbone.strides[fpn_from_stage]
        self.strides = [first * 2**i for i in range(n_levels)]  # cada nivel extra reduce a la mitad
        if box_loss not in ("smooth_l1", "iou", "giou"):
            raise ValueError(f"box_loss desconocida: {box_loss}")
        if box_loss == "smooth_l1" and smooth_l1_beta is None:
            raise ValueError("smooth_l1 requiere smooth_l1_beta")
        self.box_loss = box_loss
        if len(size_ranges) != n_levels:
            raise ValueError(f"size_ranges debe tener {n_levels} rangos, uno por nivel")
        self.size_ranges = size_ranges
        self.center_radius = center_radius
        self.use_centerness = use_centerness
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.cls_weight, self.box_weight = cls_weight, box_weight
        self.nms_iou = nms_iou
        self.score_threshold = 0.05

        self.transform = GeneralizedRCNNTransform(min_size, max_size, IMAGENET_MEAN, IMAGENET_STD)
        self.fpn = FPN(selected, fpn_channels, fpn_extra_levels, norm)
        self.head = Head(fpn_channels, nc, head_convs, n_levels, norm, use_centerness)

    def forward(self, images, targets=None):
        original_sizes = [tuple(i.shape[-2:]) for i in images]
        batch, targets = self.transform(list(images), targets)
        features = self.fpn(self.backbone(batch.tensors)[self.fpn_from_stage :])
        cls_all, reg_all, ctr_all = self.head(features)
        points = [grid_points(f, s) for f, s in zip(features, self.strides)]
        if self.training:
            return self._loss(cls_all, reg_all, ctr_all, points, targets)
        detections = self._inference(cls_all, reg_all, ctr_all, points, batch.image_sizes)
        return self.transform.postprocess(detections, batch.image_sizes, original_sizes)

    def _loss(self, cls_all, reg_all, ctr_all, points, targets):
        cls_logits, reg_preds, ctr_logits, all_labels, all_ltrb = [], [], [], [], []
        for level, (cls, reg, ctr, pts) in enumerate(zip(cls_all, reg_all, ctr_all, points)):
            stride = self.strides[level]
            for b, target in enumerate(targets):
                labels, ltrb = assign_targets(
                    pts, target["boxes"], target["labels"], stride, self.size_ranges[level], self.center_radius, self.nc
                )
                all_labels.append(labels)
                all_ltrb.append(ltrb / stride)  # misma unidad que la prediccion
                cls_logits.append(cls[b].permute(1, 2, 0).reshape(-1, self.nc))
                reg_preds.append(reg[b].permute(1, 2, 0).reshape(-1, 4))
                if ctr is not None:
                    ctr_logits.append(ctr[b].reshape(-1))

        cls_logits = torch.cat(cls_logits)
        reg_preds = torch.cat(reg_preds)
        labels = torch.cat(all_labels)
        ltrb = torch.cat(all_ltrb)
        positive = labels < self.nc
        n_positive = max(positive.sum().item(), 1)

        cls_target = torch.zeros_like(cls_logits)
        cls_target[positive, labels[positive]] = 1.0
        losses = {
            "cls": self.cls_weight
            * sigmoid_focal_loss(cls_logits, cls_target, self.focal_alpha, self.focal_gamma, reduction="sum")
            / n_positive
        }
        if positive.any():
            if self.box_loss == "smooth_l1":
                box = F.smooth_l1_loss(reg_preds[positive], ltrb[positive], beta=self.smooth_l1_beta, reduction="sum")
            else:
                box = iou_loss(reg_preds[positive], ltrb[positive], self.box_loss).sum()
        else:
            box = reg_preds.sum() * 0.0
        losses["box"] = self.box_weight * box / n_positive
        if self.use_centerness:
            logits = torch.cat(ctr_logits)
            if positive.any():
                losses["centerness"] = (
                    F.binary_cross_entropy_with_logits(
                        logits[positive], centerness_target(ltrb[positive]), reduction="sum"
                    )
                    / n_positive
                )
            else:
                losses["centerness"] = logits.sum() * 0.0
        return losses

    def _inference(self, cls_all, reg_all, ctr_all, points, image_sizes):
        results = []
        for b, (height, width) in enumerate(image_sizes):
            boxes_all, scores_all, labels_all = [], [], []
            for level, (cls, reg, ctr, pts) in enumerate(zip(cls_all, reg_all, ctr_all, points)):
                scores = cls[b].permute(1, 2, 0).reshape(-1, self.nc).sigmoid()
                if ctr is not None:
                    scores = torch.sqrt(scores * ctr[b].reshape(-1, 1).sigmoid())
                keep = scores > self.score_threshold
                values = scores[keep]
                indices = keep.nonzero()
                if len(values) > TOPK_PER_LEVEL:
                    values, top = values.topk(TOPK_PER_LEVEL)
                    indices = indices[top]
                point_index, labels = indices[:, 0], indices[:, 1]
                distances = reg[b].permute(1, 2, 0).reshape(-1, 4)[point_index] * self.strides[level]
                p = pts[point_index]
                boxes = torch.stack(
                    [
                        (p[:, 0] - distances[:, 0]).clamp(0, width),
                        (p[:, 1] - distances[:, 1]).clamp(0, height),
                        (p[:, 0] + distances[:, 2]).clamp(0, width),
                        (p[:, 1] + distances[:, 3]).clamp(0, height),
                    ],
                    dim=1,
                )
                boxes_all.append(boxes)
                scores_all.append(values)
                labels_all.append(labels)
            boxes = torch.cat(boxes_all)
            scores = torch.cat(scores_all)
            labels = torch.cat(labels_all)
            keep = batched_nms(boxes, scores, labels, self.nms_iou)[:100]
            results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        return results

    def set_score_threshold(self, conf):
        self.score_threshold = conf

    def configure_trainable(self, phase):
        for p in self.parameters():
            p.requires_grad = True

    def trainability_report(self):
        report = {}
        for name, module in (("backbone", self.backbone), ("fpn", self.fpn), ("head", self.head)):
            params = list(module.parameters())
            report[name] = {
                "params": sum(p.numel() for p in params),
                "trainable": sum(p.numel() for p in params if p.requires_grad),
            }
        return report
