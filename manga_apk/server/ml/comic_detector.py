"""
Comic Text Detector wrapper around mayocream/comic-text-detector (YOLOv5 + UNet + DBNet).
Detects speech bubbles, dark narration boxes, and borderless floating text blocks
independently of reading order, and produces a UNet binary text mask for cleaning.
"""
from copy import deepcopy
import logging
import math
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

try:
    import pyclipper
    from shapely.geometry import Polygon
except ImportError:
    pyclipper = None
    Polygon = None

try:
    from detection import (
        expand_to_bubble_contour,
        get_white_cc_labels,
        detect_stepped_box_split_x,
        detect_stepped_box_split_y,
        split_joined_bubble_lobes,
        _crop_reading_key,
    )
except (ImportError, ValueError):
    try:
        from .detection import (
            expand_to_bubble_contour,
            get_white_cc_labels,
            detect_stepped_box_split_x,
            detect_stepped_box_split_y,
            split_joined_bubble_lobes,
            _crop_reading_key,
        )
    except (ImportError, ValueError):
        try:
            from ml.detection import (
                expand_to_bubble_contour,
                get_white_cc_labels,
                detect_stepped_box_split_x,
                detect_stepped_box_split_y,
                split_joined_bubble_lobes,
                _crop_reading_key,
            )
        except (ImportError, ValueError):
            expand_to_bubble_contour = None
            get_white_cc_labels = None
            detect_stepped_box_split_x = None
            detect_stepped_box_split_y = None
            split_joined_bubble_lobes = None
            _crop_reading_key = None

try:
    from geometry import has_pinched_bubble_waist
except (ImportError, ValueError):
    try:
        from .geometry import has_pinched_bubble_waist
    except (ImportError, ValueError):
        try:
            from ml.geometry import has_pinched_bubble_waist
        except (ImportError, ValueError):
            has_pinched_bubble_waist = None

try:
    from bubble_rtdetr import get_shared_bubble_rtdetr
except (ImportError, ValueError):
    try:
        from .bubble_rtdetr import get_shared_bubble_rtdetr
    except (ImportError, ValueError):
        try:
            from ml.bubble_rtdetr import get_shared_bubble_rtdetr
        except (ImportError, ValueError):
            get_shared_bubble_rtdetr = None

logger = logging.getLogger(__name__)

TEXTDET_MASK = 0
TEXTDET_DET = 1
TEXTDET_INFERENCE = 2

LANG_LIST = ["eng", "ja", "unknown"]

_YOLOV5S_CTD_CFG = {
    "nc": 2,
    "depth_multiple": 0.33,
    "width_multiple": 0.50,
    "anchors": [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ],
    "backbone": [
        [-1, 1, "Conv", [64, 6, 2, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 3, "C3", [128]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 6, "C3", [256]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 9, "C3", [512]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 3, "C3", [1024]],
        [-1, 1, "SPPF", [1024, 5]],
    ],
    "head": [
        [-1, 1, "Conv", [512, 1, 1]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 3, "C3", [512, False]],
        [-1, 1, "Conv", [256, 1, 1]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 3, "C3", [256, False]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 14], 1, "Concat", [1]],
        [-1, 3, "C3", [512, False]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 3, "C3", [1024, False]],
        [[17, 20, 23], 1, "Detect", ["nc", "anchors"]],
    ],
}


# ---------------------------------------------------------------------------
# Minimal YOLOv5s + UNet + DBNet architecture matching comic-text-detector
# ---------------------------------------------------------------------------

def _autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def _make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def _fuse_conv_and_bn(conv, bn):
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if isinstance(act, bool):
            self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        elif isinstance(act, str):
            if act == "leaky":
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif act == "relu":
                self.act = nn.ReLU(inplace=True)
            else:
                self.act = nn.Identity()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=act)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    stride = None
    onnx_dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        dtype = self.anchors[i].dtype
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing="ij")
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).to(dtype)
        anchor_grid = (
            (self.anchors[i].clone() * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand((1, self.na, ny, nx, 2))
            .to(dtype)
        )
        return grid, anchor_grid


def _parse_yolov5_model(d, ch):
    anchors, nc, gd, gw = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"]
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, c2 = [], [], ch[-1]
    module_map = {
        "Conv": Conv,
        "C3": C3,
        "SPPF": SPPF,
        "Concat": Concat,
        "Detect": Detect,
        "nn.Upsample": nn.Upsample,
    }
    env_vars = {"nc": nc, "anchors": anchors}

    for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):
        m = module_map[m_name] if isinstance(m_name, str) else m_name
        parsed_args = []
        for a in args:
            if isinstance(a, str) and a in env_vars:
                parsed_args.append(env_vars[a])
            else:
                parsed_args.append(a)
        args = parsed_args

        n = max(round(n * gd), 1) if n > 1 else n
        if m in (Conv, C3, SPPF):
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = _make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m is C3:
                args.insert(2, n)
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        m_.i, m_.f = i, f
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class YoloV5Backbone(nn.Module):
    def __init__(self, cfg=None, ch=3):
        super().__init__()
        self.out_indices = None
        self.yaml = deepcopy(cfg or _YOLOV5S_CTD_CFG)
        ch = self.yaml.get("ch", ch)
        self.model, self.save = _parse_yolov5_model(deepcopy(self.yaml), ch=[ch])
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            a = m.anchors.prod(-1).view(-1)
            if (a[-1] - a[0]).sign() != (m.stride[-1] - m.stride[0]).sign():
                m.anchors[:] = m.anchors.flip(0)
            self.stride = m.stride

    def fuse(self):
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, "bn"):
                m.conv = _fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
        return self

    def forward(self, x, detect=False):
        y, z = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if self.out_indices is not None and m.i in self.out_indices:
                z.append(x)
        if self.out_indices is not None:
            return (x, z) if detect else z
        return x


class _DoubleConvUpC3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            C3(in_ch + mid_ch, mid_ch, act=act),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _DoubleConvC3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act=True):
        super().__init__()
        self.down = nn.AvgPool2d(2, stride=2) if stride > 1 else None
        self.conv = C3(in_ch, out_ch, act=act)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        return self.conv(x)


class UnetHead(nn.Module):
    def __init__(self, act="leaky"):
        super().__init__()
        self.down_conv1 = _DoubleConvC3(512, 512, 2, act=act)
        self.upconv0 = _DoubleConvUpC3(0, 512, 256, act=act)
        self.upconv2 = _DoubleConvUpC3(256, 512, 256, act=act)
        self.upconv3 = _DoubleConvUpC3(0, 512, 256, act=act)
        self.upconv4 = _DoubleConvUpC3(128, 256, 128, act=act)
        self.upconv5 = _DoubleConvUpC3(64, 128, 64, act=act)
        self.upconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f160, f80, f40, f20, f3, forward_mode=TEXTDET_INFERENCE):
        d10 = self.down_conv1(f3)
        u20 = self.upconv0(d10)
        u40 = self.upconv2(torch.cat([f20, u20], dim=1))
        if forward_mode == TEXTDET_DET:
            return f80, f40, u40
        u80 = self.upconv3(torch.cat([f40, u40], dim=1))
        u160 = self.upconv4(torch.cat([f80, u80], dim=1))
        u320 = self.upconv5(torch.cat([f160, u160], dim=1))
        mask = self.upconv6(u320)
        if forward_mode == TEXTDET_MASK:
            return mask
        return mask, [f80, f40, u40]


class DBHead(nn.Module):
    def __init__(self, in_channels=64, k=50, act="leaky"):
        super().__init__()
        self.k = k
        self.upconv3 = _DoubleConvUpC3(0, 512, 256, act=act)
        self.upconv4 = _DoubleConvUpC3(128, 256, 128, act=act)
        self.conv = nn.Sequential(
            nn.Conv2d(128, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
        )
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, f80, f40, u40):
        u80 = self.upconv3(torch.cat([f40, u40], dim=1))
        x = self.upconv4(torch.cat([f80, u80], dim=1))
        x = self.conv(x)
        threshold_maps = self.thresh(x)
        shrink_maps = torch.sigmoid(self.binarize(x))
        return torch.cat((shrink_maps, threshold_maps), dim=1)


class TextDetNet(nn.Module):
    def __init__(self, blk_det: YoloV5Backbone, text_seg: UnetHead, text_det: DBHead):
        super().__init__()
        self.blk_det = blk_det
        self.text_seg = text_seg
        self.text_det = text_det

    def fuse(self):
        self.blk_det.fuse()
        for mod in (self.text_seg, self.text_det):
            for m in mod.modules():
                if type(m) is Conv and hasattr(m, "bn"):
                    m.conv = _fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
        return self

    def forward(self, x):
        blks, features = self.blk_det(x, detect=True)
        mask, seg_features = self.text_seg(*features, forward_mode=TEXTDET_INFERENCE)
        lines = self.text_det(*seg_features)
        return blks[0], mask, lines


# ---------------------------------------------------------------------------
# Post-processing (NMS, DBNet contour unclip, line->block grouping)
# ---------------------------------------------------------------------------

def _letterbox(im: np.ndarray, new_shape=(1024, 1024), color=(0, 0, 0), stride=64):
    shape = im.shape[:2]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = int(new_shape[1] - new_unpad[0]), int(new_shape[0] - new_unpad[1])
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _non_max_suppression(prediction: torch.Tensor, conf_thres=0.4, iou_thres=0.35, max_det=300, agnostic: bool = True):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 4096
    max_nms = 30000
    multi_label = nc > 1
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = _xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = 0 if agnostic else (x[:, 5:6] * max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        idx = torchvision.ops.nms(boxes.float(), scores.float(), iou_thres)
        if idx.shape[0] > max_det:
            idx = idx[:max_det]
        output[xi] = x[idx]
    return output


class _SegDetectorRepresenter:
    def __init__(self, thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred: torch.Tensor):
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[1], pred.shape[2]
            boxes, scores = self._boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def _boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap.detach().cpu().numpy() if isinstance(_bitmap, torch.Tensor) else _bitmap
        pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred
        if pred_np.dtype == np.float16:
            pred_np = pred_np.astype(np.float32)
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            if contour.ndim != 2 or len(contour) < 3:
                continue
            points, sside = self._get_mini_boxes(contour)
            if sside < 2:
                continue
            score = self._box_score_fast(pred_np, contour)
            expanded = self._unclip(np.array(points), unclip_ratio=self.unclip_ratio)
            if expanded.size == 0:
                continue
            box, sside = self._get_mini_boxes(expanded.reshape(-1, 1, 2))
            box = np.array(box, dtype=np.float32)
            box[:, 0] = np.clip(np.round(box[:, 0] / max(1, width) * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / max(1, height) * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int32))
            scores.append(score)

        if not boxes:
            return np.zeros((0, 4, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
        return np.array(boxes, dtype=np.int32), np.array(scores, dtype=np.float32)

    @staticmethod
    def _unclip(box: np.ndarray, unclip_ratio=1.5) -> np.ndarray:
        if Polygon is not None and pyclipper is not None:
            poly = Polygon(box)
            if poly.length <= 0 or poly.area <= 0:
                return np.array([])
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box.astype(np.int32).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            res = offset.Execute(distance)
            if not res:
                return np.array([])
            return np.array(res[0], dtype=np.float32)
        # Lightweight fallback when pyclipper/shapely are absent
        cx, cy = box.mean(axis=0)
        return (box - [cx, cy]) * 1.25 + [cx, cy]

    @staticmethod
    def _get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour.astype(np.float32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            i1, i4 = 0, 1
        else:
            i1, i4 = 1, 0
        if points[3][1] > points[2][1]:
            i2, i3 = 2, 3
        else:
            i2, i3 = 3, 2
        return [points[i1], points[i2], points[i3], points[i4]], min(bounding_box[1])

    @staticmethod
    def _box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])


def _union_area(bboxa, bboxb) -> float:
    x1 = max(bboxa[0], bboxb[0])
    y1 = max(bboxa[1], bboxb[1])
    x2 = min(bboxa[2], bboxb[2])
    y2 = min(bboxa[3], bboxb[3])
    if y2 < y1 or x2 < x1:
        return -1.0
    return float((y2 - y1) * (x2 - x1))


def _rotate_polygons(center, polygons, rotation, to_int=True):
    rad = np.deg2rad(rotation)
    s, c = np.sin(rad), np.cos(rad)
    polygons = polygons.astype(np.float32).copy()
    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += center[1]
    rotated[:, ::2] += center[0]
    return rotated.astype(np.int64) if to_int else rotated


class _CTDBlock:
    def __init__(self, xyxy: List[int], lines=None, language="unknown", conf: float = 1.0):
        self.xyxy = [int(v) for v in xyxy]
        self.lines = [] if lines is None else list(lines)
        self.language = language
        self.conf = float(conf)
        self.vertical = False
        self.font_size = -1.0
        self.distance = None
        self.angle = 0
        self.vec = None
        self.norm = -1.0
        self.merged = False

    def center(self):
        arr = np.array(self.xyxy, dtype=np.float64)
        return (arr[:2] + arr[2:]) / 2.0

    def adjust_bbox(self, with_bbox=False):
        if not self.lines:
            return
        lines = np.array(self.lines, dtype=np.int32)
        if with_bbox:
            self.xyxy[0] = min(int(lines[..., 0].min()), self.xyxy[0])
            self.xyxy[1] = min(int(lines[..., 1].min()), self.xyxy[1])
            self.xyxy[2] = max(int(lines[..., 0].max()), self.xyxy[2])
            self.xyxy[3] = max(int(lines[..., 1].max()), self.xyxy[3])
        else:
            self.xyxy[0] = int(lines[..., 0].min())
            self.xyxy[1] = int(lines[..., 1].min())
            self.xyxy[2] = int(lines[..., 0].max())
            self.xyxy[3] = int(lines[..., 1].max())

    def sort_lines(self):
        if self.distance is not None and len(self.lines) > 1:
            idx = np.argsort(self.distance)
            self.distance = self.distance[idx]
            lines = np.array(self.lines, dtype=np.int32)
            self.lines = lines[idx].tolist()


def _examine_blk(blk: _CTDBlock, im_w: int, sort: bool = False):
    lines = np.array(blk.lines, dtype=np.float64)
    if lines.size == 0:
        return
    middle_pnts = (lines[:, [1, 2, 3, 0]] + lines) / 2.0
    vec_v = middle_pnts[:, 2] - middle_pnts[:, 0]
    vec_h = middle_pnts[:, 1] - middle_pnts[:, 3]
    center_pnts = (lines[:, 0] + lines[:, 2]) / 2.0
    v = np.sum(vec_v, axis=0)
    h = np.sum(vec_h, axis=0)
    norm_v, norm_h = float(np.linalg.norm(v)), float(np.linalg.norm(h))
    vertical = (norm_v > norm_h) if blk.language == "ja" else (norm_v > norm_h * 2.0)

    if vertical:
        primary_vec, primary_norm = v, max(norm_v, 1e-6)
        distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)
        font_size = max(1.0, round(norm_h / len(lines)))
    else:
        primary_vec, primary_norm = h, max(norm_h, 1e-6)
        distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
        font_size = max(1.0, round(norm_v / len(lines)))

    rotation_angle = int(math.atan2(primary_vec[1], primary_vec[0]) / math.pi * 180.0)
    dist = np.linalg.norm(distance_vectors, axis=1)
    denom = np.maximum(dist * primary_norm, 1e-6)
    cos_val = np.clip(np.einsum("ij, j->i", distance_vectors, primary_vec) / denom, -1.0, 1.0)
    rad_matrix = np.arccos(cos_val)
    distance = np.abs(np.sin(rad_matrix) * dist)

    blk.lines = lines.astype(np.int32).tolist()
    blk.distance = distance
    blk.angle = rotation_angle - 90 if vertical else rotation_angle
    if abs(blk.angle) < 3:
        blk.angle = 0
    blk.font_size = float(font_size)
    blk.vertical = vertical
    blk.vec = primary_vec
    blk.norm = float(primary_norm)
    if sort:
        blk.sort_lines()


def _merge_scattered_lines(blk_list: List[_CTDBlock]) -> List[_CTDBlock]:
    if len(blk_list) < 2:
        return blk_list
    blk_list.sort(key=lambda b: b.distance[0] if b.distance is not None else 0.0)
    merged_list = []
    for ii, cur in enumerate(blk_list):
        if cur.merged:
            continue
        for blk2 in blk_list[ii + 1:]:
            if blk2.merged or cur.font_size <= 0 or blk2.font_size <= 0 or cur.norm <= 0 or blk2.norm <= 0:
                continue
            fnt_div = cur.font_size / blk2.font_size
            num1, num2 = len(cur.lines), len(blk2.lines)
            fnt_avg = (cur.font_size * num1 + blk2.font_size * num2) / max(1, num1 + num2)
            cos_vec = float((cur.vec @ blk2.vec) / max(1e-6, cur.norm * blk2.norm))
            distance = float(blk2.distance[-1] - cur.distance[-1])
            distance_p1 = float(np.linalg.norm(np.array(blk2.lines[-1][0]) - np.array(cur.lines[-1][0])))

            intersects = False
            if Polygon is not None:
                try:
                    intersects = Polygon(cur.lines[-1]).intersects(Polygon(blk2.lines[-1]))
                except Exception:
                    intersects = False
            if not intersects:
                if fnt_div > 1.3 or (1.0 / max(1e-6, fnt_div)) > 1.3:
                    continue
                if abs(cos_vec) < 0.866:
                    continue
                if distance > 2.0 * fnt_avg or distance_p1 > fnt_avg * 2.5:
                    continue
            cur.lines.append(blk2.lines[0])
            vec_sum = cur.vec + blk2.vec
            cur.vec = vec_sum
            cur.angle = int(round(np.rad2deg(math.atan2(vec_sum[1], vec_sum[0]))))
            if cur.vertical:
                cur.angle -= 90
            cur.norm = float(np.linalg.norm(vec_sum))
            cur.distance = np.append(cur.distance, blk2.distance[-1])
            cur.font_size = fnt_avg
            blk2.merged = True
        merged_list.append(cur)
    for blk in merged_list:
        blk.adjust_bbox(with_bbox=False)
    return merged_list


def _split_blk_on_gap(blk: _CTDBlock) -> Tuple[bool, List[_CTDBlock]]:
    if len(blk.lines) <= 1:
        return False, [blk]
    font_size, distance, lines = blk.font_size, blk.distance, blk.lines
    l0 = np.array(lines[0])
    if distance is not None and len(distance) == len(lines):
        paired = sorted(zip(lines, distance), key=lambda p: np.linalg.norm(np.array(p[0][0]) - l0[0]))
        lines = [p[0] for p in paired]
        distance = [p[1] for p in paired]
        blk.lines = lines
        blk.distance = distance
    else:
        lines.sort(key=lambda line: np.linalg.norm(np.array(line[0]) - l0[0]))
    distance_tol = max(10.0, font_size * 2.0)
    current_blk = deepcopy(blk)
    current_blk.lines = [lines[0]]
    sub_blk_list = [current_blk]
    for jj, line in enumerate(lines[1:]):
        intersects = False
        if Polygon is not None:
            try:
                intersects = Polygon(lines[jj]).intersects(Polygon(line))
            except Exception:
                intersects = False
        split = False
        if not intersects and distance is not None and jj + 1 < len(distance):
            line_dist = abs(distance[jj + 1] - distance[jj])
            if line_dist > distance_tol:
                split = True
            elif blk.vertical and abs(blk.angle) < 15:
                if len(current_blk.lines) > 1 or line_dist > font_size:
                    split = abs(lines[jj][0][1] - line[0][1]) > font_size
        if split:
            current_blk = deepcopy(current_blk)
            current_blk.lines = [line]
            sub_blk_list.append(current_blk)
        else:
            current_blk.lines.append(line)
    if len(sub_blk_list) > 1:
        for sb in sub_blk_list:
            sb.adjust_bbox(with_bbox=False)
        return True, sub_blk_list
    return False, sub_blk_list


def _split_horizontal_blk_columns(
    blk: _CTDBlock,
    gray: Optional[np.ndarray] = None,
    mask_u8: Optional[np.ndarray] = None,
    depth: int = 0,
    container_box: Optional[List[int]] = None,
    bubble_box: Optional[List[int]] = None,
) -> Tuple[bool, List[_CTDBlock]]:
    """
    Separates side-by-side or linked/pinched speech bubble columns into independent blocks.
    1. Builds a 2D connectivity graph over lines (requiring horizontal overlap >= 30% and vertical gap <= 1.6*h).
    2. Enforces dark panel gutter (gray < 100 spanning >= 40%) and pinched-waist barriers.
    3. If still merged into 1 component (including 1 wide line), checks UNet projection valley / dark vertical border / stepped box boundary.
    """
    if len(blk.lines) == 0:
        return False, [blk]

    line_boxes = []
    line_hs = []
    for line in blk.lines:
        pts = np.asarray(line, dtype=np.int32)
        lx1, ly1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        lx2, ly2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        line_boxes.append([lx1, ly1, lx2, ly2])
        line_hs.append(max(1, ly2 - ly1))

    all_x1 = min(b[0] for b in line_boxes)
    all_y1 = min(b[1] for b in line_boxes)
    all_x2 = max(b[2] for b in line_boxes)
    all_y2 = max(b[3] for b in line_boxes)
    bw = all_x2 - all_x1
    bh = all_y2 - all_y1

    # A narrow text region (< 90px wide) is a single column unless separated by a tall dark vertical gutter
    if bw < 90:
        has_dark_gutter = False
        if bh >= 28 and gray is not None and gray.size > 0 and not _classify_dark_text_block(gray, [all_x1, all_y1, all_x2, all_y2]):
            gy1, gy2 = max(0, all_y1), min(gray.shape[0], all_y2)
            gx1, gx2 = max(0, all_x1), min(gray.shape[1], all_x2)
            if gy2 > gy1 and gx2 > gx1 + 10:
                mid_s = int(round(0.18 * (gx2 - gx1)))
                mid_e = int(round(0.82 * (gx2 - gx1)))
                if mid_e > mid_s:
                    dark_cols = np.mean(gray[gy1:gy2, gx1 + mid_s:gx1 + mid_e] < 100, axis=0)
                    if np.any(dark_cols >= 0.75):
                        has_dark_gutter = True
        if not has_dark_gutter:
            return False, [blk]

    median_line_h = float(np.median(line_hs)) if line_hs else 15.0
    N = len(blk.lines)

    if N >= 2:
        adj = {i: set() for i in range(N)}

        # 1. 2D Graph over blk.lines
        for i in range(N):
            b1 = line_boxes[i]
            w1 = max(1, b1[2] - b1[0])
            for j in range(i + 1, N):
                b2 = line_boxes[j]
                w2 = max(1, b2[2] - b2[0])
                h1 = max(1, b1[3] - b1[1])
                h2 = max(1, b2[3] - b2[1])
                min_w = min(w1, w2)
                min_h = min(h1, h2)
                x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                overlap_ratio = x_overlap / float(min_w)
                y_overlap_ratio = y_overlap / float(min_h)
                x_gap = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                y_gap = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))

                same_row = (y_overlap_ratio >= 0.30 and x_gap <= 0.85 * median_line_h)
                stacked_col = (overlap_ratio >= 0.15 and y_gap <= 1.6 * median_line_h)

                # Connect if same horizontal row (word space) or stacked column, unless separated by a barrier
                if same_row or stacked_col:
                    has_barrier = False
                    if gray is not None and gray.size > 0:
                        sep_x1 = min(b1[2], b2[2])
                        sep_x2 = max(b1[0], b2[0])
                        # If horizontally separated with a distinct gap
                        if sep_x2 - sep_x1 >= 2:
                            sy1 = max(0, min(b1[1], b2[1]))
                            sy2 = min(gray.shape[0], max(b1[3], b2[3]))
                            if sy2 > sy1 and sep_x2 <= gray.shape[1] and sep_x1 >= 0:
                                strip = gray[sy1:sy2, sep_x1:sep_x2]
                                if strip.size > 0:
                                    dark_cols = np.mean(strip < 100, axis=0)
                                    if np.any(dark_cols >= 0.40):
                                        has_barrier = True

                        if not has_barrier and has_pinched_bubble_waist is not None:
                            try:
                                if has_pinched_bubble_waist(gray, b1, b2):
                                    has_barrier = True
                            except Exception:
                                pass

                    if not has_barrier:
                        adj[i].add(j)
                        adj[j].add(i)

        # Find connected components
        visited = set()
        components: List[List[int]] = []
        for i in range(N):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                components.append(comp)

        if len(components) >= 2:
            # Sort components left-to-right (or reading order by horizontal center)
            components.sort(key=lambda c: float(np.mean([(line_boxes[idx][0] + line_boxes[idx][2]) / 2.0 for idx in c])))
            sub_blk_list = []
            for comp in components:
                sub = deepcopy(blk)
                sub.lines = [blk.lines[idx] for idx in comp]
                sub.adjust_bbox(with_bbox=False)
                if depth < 2:
                    _, nested = _split_horizontal_blk_columns(sub, gray=gray, mask_u8=mask_u8, depth=depth + 1)
                    sub_blk_list.extend(nested)
                else:
                    sub_blk_list.append(sub)
            return True, sub_blk_list

    # 3. UNet projection-valley split / dark vertical gutter / stepped box split
    all_x1 = min(b[0] for b in line_boxes)
    all_y1 = min(b[1] for b in line_boxes)
    all_x2 = max(b[2] for b in line_boxes)
    all_y2 = max(b[3] for b in line_boxes)
    bw = all_x2 - all_x1
    bh = all_y2 - all_y1

    if bw >= 40 and (N == 1 or bw >= int(round(2.0 * median_line_h))):
        split_x = None
        split_y = None
        eff_box = bubble_box if bubble_box is not None else container_box
        is_dark_region = (
            gray is not None
            and gray.size > 0
            and _classify_dark_text_block(gray, [all_x1, all_y1, all_x2, all_y2])
        )

        # Check stepped narration box split (horizontal X-step and vertical Y-step)
        if gray is not None:
            step_box = eff_box if eff_box is not None else [all_x1, all_y1, all_x2, all_y2]
            if detect_stepped_box_split_x is not None:
                try:
                    x_step = detect_stepped_box_split_x(gray, step_box)
                    if x_step is not None and (all_x1 + 0.15 * bw <= x_step <= all_x1 + 0.85 * bw):
                        xs_int = int(x_step)
                        has_left = False
                        has_right = False
                        if mask_u8 is not None and mask_u8.size > 0:
                            gy1, gy2 = max(0, all_y1), min(mask_u8.shape[0], all_y2)
                            gx1, gx2 = max(0, all_x1), min(mask_u8.shape[1], all_x2)
                            if gy2 > gy1 and gx1 < xs_int < gx2:
                                has_left = int(np.sum(mask_u8[gy1:gy2, gx1:xs_int] > 35)) >= 15
                                has_right = int(np.sum(mask_u8[gy1:gy2, xs_int:gx2] > 35)) >= 15
                        elif gray.size > 0:
                            gy1, gy2 = max(0, all_y1), min(gray.shape[0], all_y2)
                            gx1, gx2 = max(0, all_x1), min(gray.shape[1], all_x2)
                            if gy2 > gy1 and gx1 < xs_int < gx2:
                                if is_dark_region:
                                    med_g = float(np.median(gray[gy1:gy2, gx1:gx2]))
                                    has_left = int(np.sum(gray[gy1:gy2, gx1:xs_int] > med_g + 40)) >= 15
                                    has_right = int(np.sum(gray[gy1:gy2, xs_int:gx2] > med_g + 40)) >= 15
                                else:
                                    has_left = int(np.sum(gray[gy1:gy2, gx1:xs_int] < 165)) >= 15
                                    has_right = int(np.sum(gray[gy1:gy2, xs_int:gx2] < 165)) >= 15
                        if has_left and has_right:
                            split_x = xs_int
                except Exception:
                    pass

            if split_x is None and detect_stepped_box_split_y is not None and bh >= 40:
                try:
                    y_step = detect_stepped_box_split_y(gray, step_box)
                    if y_step is not None and (all_y1 + 0.15 * bh <= y_step <= all_y1 + 0.85 * bh):
                        ys_int = int(y_step)
                        has_top = False
                        has_bot = False
                        if mask_u8 is not None and mask_u8.size > 0:
                            gy1, gy2 = max(0, all_y1), min(mask_u8.shape[0], all_y2)
                            gx1, gx2 = max(0, all_x1), min(mask_u8.shape[1], all_x2)
                            if gx2 > gx1 and gy1 < ys_int < gy2:
                                has_top = int(np.sum(mask_u8[gy1:ys_int, gx1:gx2] > 35)) >= 15
                                has_bot = int(np.sum(mask_u8[ys_int:gy2, gx1:gx2] > 35)) >= 15
                        elif gray.size > 0:
                            gy1, gy2 = max(0, all_y1), min(gray.shape[0], all_y2)
                            gx1, gx2 = max(0, all_x1), min(gray.shape[1], all_x2)
                            if gx2 > gx1 and gy1 < ys_int < gy2:
                                if is_dark_region:
                                    med_g = float(np.median(gray[gy1:gy2, gx1:gx2]))
                                    has_top = int(np.sum(gray[gy1:ys_int, gx1:gx2] > med_g + 40)) >= 15
                                    has_bot = int(np.sum(gray[ys_int:gy2, gx1:gx2] > med_g + 40)) >= 15
                                else:
                                    has_top = int(np.sum(gray[gy1:ys_int, gx1:gx2] < 165)) >= 15
                                    has_bot = int(np.sum(gray[ys_int:gy2, gx1:gx2] < 165)) >= 15
                        if has_top and has_bot:
                            split_y = ys_int
                except Exception:
                    pass

        # Check dark vertical border in gray (only on light-background regions; dark boxes have <100 bg everywhere)
        if split_x is None and not is_dark_region and gray is not None and gray.size > 0:
            gy1 = max(0, all_y1)
            gy2 = min(gray.shape[0], all_y2)
            gx1 = max(0, all_x1)
            gx2 = min(gray.shape[1], all_x2)
            if gy2 > gy1 and gx2 > gx1:
                patch_g = gray[gy1:gy2, gx1:gx2]
                if patch_g.shape[1] == bw and patch_g.shape[0] > 0:
                    mid_s = int(round(0.18 * bw))
                    mid_e = int(round(0.82 * bw))
                    dark_col_ratios = np.mean(patch_g[:, mid_s:mid_e] < 100, axis=0)
                    dark_candidates = np.where(dark_col_ratios >= 0.40)[0]
                    if len(dark_candidates) > 0:
                        target_m = (mid_e - mid_s) / 2.0
                        best_rel = dark_candidates[np.argmin(np.abs(dark_candidates - target_m))]
                        split_x = all_x1 + mid_s + int(best_rel)

        # Check projection valley (for wide double-column bubbles or pinched-waist double bubbles)
        is_pinched_double = False
        if (
            split_x is None
            and bw >= 80
            and bw >= 0.80 * max(1, bh)
            and has_pinched_bubble_waist is not None
            and gray is not None
            and gray.size > 0
        ):
            try:
                mid_hx = (all_x1 + all_x2) // 2
                is_pinched_double = bool(
                    has_pinched_bubble_waist(
                        gray,
                        [all_x1, all_y1, mid_hx, all_y2],
                        [mid_hx, all_y1, all_x2, all_y2],
                    )
                )
            except Exception:
                is_pinched_double = False

        if split_x is None and split_y is None and ((bw >= 90 and bw >= 1.15 * max(1, bh)) or is_pinched_double):
            if mask_u8 is None and gray is None:
                return False, [blk]
            ref_h = mask_u8.shape[0] if mask_u8 is not None else gray.shape[0]
            ref_w = mask_u8.shape[1] if mask_u8 is not None else gray.shape[1]
            my1 = max(0, all_y1)
            my2 = min(ref_h, all_y2)
            mx1 = max(0, all_x1)
            mx2 = min(ref_w, all_x2)
            if my2 > my1 and mx2 > mx1:
                ink_mask = None
                if gray is not None and gray.shape[0] >= my2 and gray.shape[1] >= mx2:
                    patch_g = gray[my1:my2, mx1:mx2]
                    if patch_g.shape[1] == bw and patch_g.shape[0] > 0:
                        if is_dark_region or float(np.median(patch_g)) < 110.0:
                            cand_ink = patch_g > (float(np.median(patch_g)) + 45.0)
                            if int(np.sum(cand_ink)) >= 40:
                                ink_mask = cand_ink
                        else:
                            # Crisp dark-ink-on-light-bubble mask; intersect with mask_u8 > 20 when available to ignore bubble-border spikes
                            cand_ink = patch_g < 165
                            if mask_u8 is not None and mask_u8.shape[0] >= my2 and mask_u8.shape[1] >= mx2:
                                patch_m = mask_u8[my1:my2, mx1:mx2]
                                if patch_m.shape == cand_ink.shape and int(np.sum(patch_m > 20)) >= 40:
                                    cand_ink = np.logical_and(cand_ink, patch_m > 20)
                            if int(np.sum(cand_ink)) >= 40:
                                ink_mask = cand_ink
                if ink_mask is None and mask_u8 is not None and mask_u8.size > 0:
                    patch_m = mask_u8[my1:my2, mx1:mx2]
                    if patch_m.shape[1] == bw and patch_m.shape[0] > 0:
                        ink_mask = patch_m > 110

                if ink_mask is not None and ink_mask.shape[1] == bw and ink_mask.shape[0] > 0:
                    # Bridge normal inter-word spaces inside a single English line while keeping two-bubble gutters open
                    close_w = 5
                    ink_u8 = (ink_mask.astype(np.uint8)) * 255
                    closed_ink = cv2.morphologyEx(
                        ink_u8, cv2.MORPH_CLOSE, np.ones((1, close_w), dtype=np.uint8)
                    ) > 0

                    mid_s = int(round(0.18 * bw))
                    mid_e = int(round(0.82 * bw))
                    if mid_e > mid_s + 4:
                        col_ink = np.sum(closed_ink[:, mid_s:mid_e], axis=0)
                        is_valley = col_ink <= 1
                        best_len = 0
                        best_s = -1
                        best_e = -1
                        cur_s = -1
                        for k, val in enumerate(is_valley):
                            if val:
                                if cur_s < 0:
                                    cur_s = k
                            else:
                                if cur_s >= 0:
                                    run_len = k - cur_s
                                    if run_len > best_len:
                                        best_len = run_len
                                        best_s = cur_s
                                        best_e = k - 1
                                    cur_s = -1
                        if cur_s >= 0:
                            run_len = len(is_valley) - cur_s
                            if run_len > best_len:
                                best_len = run_len
                                best_s = cur_s
                                best_e = len(is_valley) - 1

                        if best_len > 0:
                            abs_s = mid_s + best_s
                            abs_e = mid_s + best_e
                            has_left_ink = int(np.sum(closed_ink[:, :abs_s])) >= 25
                            has_right_ink = int(np.sum(closed_ink[:, abs_e + 1:])) >= 25
                            if has_left_ink and has_right_ink:
                                min_valley_w = max(8, int(0.07 * bw))
                                cand_x = all_x1 + mid_s + (best_s + best_e) // 2
                                if best_len >= 4:
                                    allow_split = best_len >= min_valley_w
                                    if gray is not None and gray.size > 0 and not is_dark_region:
                                        # Verify dark dividing border, concave pinch, or wide clean valley
                                        wide_clean_valley = best_len >= max(8, int(0.07 * bw))
                                        nb_x1 = max(0, cand_x - 4)
                                        nb_x2 = min(gray.shape[1], cand_x + 5)
                                        has_near_border = False
                                        if nb_x2 > nb_x1 and my2 > my1:
                                            has_near_border = bool(
                                                np.any(np.mean(gray[my1:my2, nb_x1:nb_x2] < 100, axis=0) >= 0.40)
                                            )
                                        has_pinch = False
                                        if not wide_clean_valley and not has_near_border and 0 <= cand_x < gray.shape[1]:
                                            py1 = max(0, container_box[1]) if container_box is not None else my1
                                            py2 = min(gray.shape[0], container_box[3]) if container_box is not None else my2
                                            if py2 > py1 + 4:
                                                mid_white = _max_contiguous_true(gray[py1:py2, cand_x] > 170)
                                                step_x = max(1, bw // 16)
                                                left_xs = range(mx1, max(mx1 + 1, all_x1 + abs_s), step_x)
                                                right_xs = range(min(mx2 - 1, all_x1 + abs_e + 1), mx2, step_x)
                                                left_white = max(
                                                    (_max_contiguous_true(gray[py1:py2, x] > 170) for x in left_xs),
                                                    default=0,
                                                )
                                                right_white = max(
                                                    (_max_contiguous_true(gray[py1:py2, x] > 170) for x in right_xs),
                                                    default=0,
                                                )
                                                taller_side = max(left_white, right_white)
                                                if taller_side > 0 and mid_white < 0.78 * float(taller_side):
                                                    has_pinch = True
                                        allow_split = (
                                            (best_len >= min_valley_w and (wide_clean_valley or has_near_border or has_pinch))
                                            or (has_pinch and best_len >= 4)
                                        )
                                    if allow_split:
                                        split_x = cand_x

        if split_x is not None:
            left_lines = []
            right_lines = []
            for idx, line in enumerate(blk.lines):
                lx1, ly1, lx2, ly2 = line_boxes[idx]
                if lx2 <= split_x:
                    left_lines.append(line)
                elif lx1 >= split_x:
                    right_lines.append(line)
                else:
                    # Line polygon crosses split_x: slice into left and right rectangles
                    if split_x - 1 > lx1 + 2:
                        left_lines.append(
                            np.array([[lx1, ly1], [split_x - 1, ly1], [split_x - 1, ly2], [lx1, ly2]], dtype=np.int32)
                        )
                    if lx2 > split_x + 3:
                        right_lines.append(
                            np.array([[split_x + 1, ly1], [lx2, ly1], [lx2, ly2], [split_x + 1, ly2]], dtype=np.int32)
                        )

            if left_lines and right_lines:
                sub_blks: List[_CTDBlock] = []
                for side_lines, side_min_x, side_max_x in [
                    (left_lines, all_x1, split_x),
                    (right_lines, split_x, all_x2),
                ]:
                    sub = deepcopy(blk)
                    sub.lines = side_lines
                    sub.adjust_bbox(with_bbox=False)
                    # Tighten side xyxy to actual mask_u8 > 35 ink bounding box on its side of split_x
                    if mask_u8 is not None and mask_u8.size > 0:
                        sy1 = max(0, int(sub.xyxy[1]))
                        sy2 = min(mask_u8.shape[0], int(sub.xyxy[3]))
                        sx1 = max(0, max(int(sub.xyxy[0]), side_min_x))
                        sx2 = min(mask_u8.shape[1], min(int(sub.xyxy[2]), side_max_x))
                        if sy2 > sy1 and sx2 > sx1:
                            ys, xs = np.where(mask_u8[sy1:sy2, sx1:sx2] > 35)
                            if xs.size > 0:
                                sub.xyxy = [
                                    sx1 + int(xs.min()),
                                    sy1 + int(ys.min()),
                                    sx1 + int(xs.max()) + 1,
                                    sy1 + int(ys.max()) + 1,
                                ]
                    if depth < 2:
                        _, nested = _split_horizontal_blk_columns(sub, gray=gray, mask_u8=mask_u8, depth=depth + 1)
                        sub_blks.extend(nested)
                    else:
                        sub_blks.append(sub)
                if len(sub_blks) >= 2:
                    return True, sub_blks

        if split_y is not None:
            top_lines = []
            bot_lines = []
            for idx, line in enumerate(blk.lines):
                lx1, ly1, lx2, ly2 = line_boxes[idx]
                if ly2 <= split_y:
                    top_lines.append(line)
                elif ly1 >= split_y:
                    bot_lines.append(line)
                else:
                    if split_y - 1 > ly1 + 2:
                        top_lines.append(
                            np.array([[lx1, ly1], [lx2, ly1], [lx2, split_y - 1], [lx1, split_y - 1]], dtype=np.int32)
                        )
                    if ly2 > split_y + 3:
                        bot_lines.append(
                            np.array([[lx1, split_y + 1], [lx2, split_y + 1], [lx2, ly2], [lx1, ly2]], dtype=np.int32)
                        )

            if top_lines and bot_lines:
                sub_blks_y: List[_CTDBlock] = []
                for side_lines, side_min_y, side_max_y in [
                    (top_lines, all_y1, split_y),
                    (bot_lines, split_y, all_y2),
                ]:
                    sub = deepcopy(blk)
                    sub.lines = side_lines
                    sub.adjust_bbox(with_bbox=False)
                    if mask_u8 is not None and mask_u8.size > 0:
                        sy1 = max(0, max(int(sub.xyxy[1]), side_min_y))
                        sy2 = min(mask_u8.shape[0], min(int(sub.xyxy[3]), side_max_y))
                        sx1 = max(0, int(sub.xyxy[0]))
                        sx2 = min(mask_u8.shape[1], int(sub.xyxy[2]))
                        if sy2 > sy1 and sx2 > sx1:
                            ys, xs = np.where(mask_u8[sy1:sy2, sx1:sx2] > 35)
                            if xs.size > 0:
                                sub.xyxy = [
                                    sx1 + int(xs.min()),
                                    sy1 + int(ys.min()),
                                    sx1 + int(xs.max()) + 1,
                                    sy1 + int(ys.max()) + 1,
                                ]
                    if depth < 2:
                        _, nested = _split_horizontal_blk_columns(sub, gray=gray, mask_u8=mask_u8, depth=depth + 1)
                        sub_blks_y.extend(nested)
                    else:
                        sub_blks_y.append(sub)
                if len(sub_blks_y) >= 2:
                    return True, sub_blks_y

    return False, [blk]


def _group_output(
    blks, lines, im_w: int, im_h: int, mask: Optional[np.ndarray], gray: Optional[np.ndarray] = None
) -> List[_CTDBlock]:
    blk_list: List[_CTDBlock] = []
    scattered_lines = {"ver": [], "hor": []}
    for bbox, cls, conf in zip(*blks):
        lang = LANG_LIST[int(cls)] if 0 <= int(cls) < len(LANG_LIST) else "unknown"
        blk_list.append(_CTDBlock(bbox, language=lang, conf=float(conf)))

    bbox_score_thresh = 0.4
    mask_score_thresh = 0.20
    for line in lines:
        bx1, bx2 = int(line[:, 0].min()), int(line[:, 0].max())
        by1, by2 = int(line[:, 1].min()), int(line[:, 1].max())
        line_area = max(1, (by2 - by1) * (bx2 - bx1))
        bbox_score, bbox_idx = -1.0, -1
        for jj, blk in enumerate(blk_list):
            score = _union_area(blk.xyxy, [bx1, by1, bx2, by2]) / float(line_area)
            if bbox_score < score:
                bbox_score = score
                bbox_idx = jj
        if bbox_score > bbox_score_thresh and bbox_idx >= 0:
            blk_list[bbox_idx].lines.append(line)
        else:
            if mask is not None:
                patch_m = mask[max(0, by1):min(im_h, by2), max(0, bx1):min(im_w, bx2)]
                if patch_m.size == 0 or (float(patch_m.mean()) / 255.0) < mask_score_thresh:
                    continue
            blk = _CTDBlock([bx1, by1, bx2, by2], [line], conf=0.75)
            _examine_blk(blk, im_w, sort=False)
            if blk.vertical:
                scattered_lines["ver"].append(blk)
            else:
                scattered_lines["hor"].append(blk)

    final_blk_list: List[_CTDBlock] = []
    for blk in blk_list:
        if len(blk.lines) == 0:
            bx1, by1, bx2, by2 = blk.xyxy
            if mask is not None:
                patch_m = mask[max(0, by1):min(im_h, by2), max(0, bx1):min(im_w, bx2)]
                if patch_m.size == 0 or (float(patch_m.mean()) / 255.0) < mask_score_thresh:
                    continue
            blk.lines = [
                [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
            ]
        _examine_blk(blk, im_w, sort=True)
        splitted = False
        if len(blk.lines) >= 1:
            if blk.language == "ja" or blk.vertical:
                splitted, sub_blk_list = _split_blk_on_gap(blk)
            else:
                splitted, sub_blk_list = _split_horizontal_blk_columns(blk, gray, mask)
        else:
            sub_blk_list = [blk]
        if not splitted:
            for sb in sub_blk_list:
                sb.adjust_bbox(with_bbox=True)
        else:
            for sb in sub_blk_list:
                _examine_blk(sb, im_w, sort=True)
                sb.adjust_bbox(with_bbox=False)
        final_blk_list.extend(sub_blk_list)

    final_blk_list.extend(_merge_scattered_lines(scattered_lines["hor"]))
    final_blk_list.extend(_merge_scattered_lines(scattered_lines["ver"]))
    return final_blk_list


def _classify_dark_text_block(gray: np.ndarray, tbox: List[int]) -> bool:
    """Return True only when the text box background is genuinely dark with light text strokes."""
    im_h, im_w = gray.shape[:2]
    tx1, ty1, tx2, ty2 = tbox
    px1, py1 = max(0, tx1 - 6), max(0, ty1 - 6)
    px2, py2 = min(im_w, tx2 + 6), min(im_h, ty2 + 6)
    patch = gray[py1:py2, px1:px2]
    t_patch = gray[ty1:ty2, tx1:tx2]
    if patch.size == 0 or t_patch.size == 0:
        return False

    med_all = float(np.median(patch))
    med_t = float(np.median(t_patch))
    p95_t = float(np.percentile(t_patch, 95))
    p95_p = float(np.percentile(patch, 95))
    p98_p = float(np.percentile(patch, 98))
    has_bright = (p95_p > 170) or (p98_p > 175 and int(np.sum(patch > 170)) >= 8)

    return bool(
        (med_all < 110 and med_t < 120 and has_bright)
        or (med_all < 130 and med_t < 120 and p95_t > 175 and (p95_t - med_t) > 65)
    )


def _expand_dark_narration_box(gray: np.ndarray, text_box: List[int], other_tboxes: List[List[int]], max_pad: int = 40) -> List[int]:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [max(0, text_box[0] - 4), max(0, text_box[1] - 4), min(w, text_box[2] + 4), min(h, text_box[3] + 4)]
    lim_l, lim_r, lim_t, lim_b = 0, w - 1, 0, h - 1
    for ox1, oy1, ox2, oy2 in other_tboxes:
        if max(y1, oy1) < min(y2, oy2):
            if ox2 <= x1:
                lim_l = max(lim_l, (ox2 + x1) // 2 + 1)
            elif ox1 >= x2:
                lim_r = min(lim_r, (x2 + ox1) // 2 - 1)
        if max(x1, ox1) < min(x2, ox2):
            if oy2 <= y1:
                lim_t = max(lim_t, (oy2 + y1) // 2 + 1)
            elif oy1 >= y2:
                lim_b = min(lim_b, (y2 + oy1) // 2 - 1)

    cx1, cx2, cy1, cy2 = x1, x2, y1, y2
    for d in range(1, max_pad):
        tx = x1 - d
        if tx < lim_l:
            break
        col = gray[y1:y2, tx]
        if col.size == 0 or np.mean(col > 170) >= 0.40 or np.mean(col) > 150:
            break
        cx1 = tx
    for d in range(1, max_pad):
        tx = x2 + d - 1
        if tx > lim_r:
            break
        col = gray[y1:y2, tx]
        if col.size == 0 or np.mean(col > 170) >= 0.40 or np.mean(col) > 150:
            break
        cx2 = tx + 1
    for d in range(1, max_pad):
        ty = y1 - d
        if ty < lim_t:
            break
        row = gray[ty, cx1:cx2]
        if row.size == 0 or np.mean(row > 170) >= 0.40 or np.mean(row) > 150:
            break
        cy1 = ty
    for d in range(1, max_pad):
        ty = y2 + d - 1
        if ty > lim_b:
            break
        row = gray[ty, cx1:cx2]
        if row.size == 0 or np.mean(row > 170) >= 0.40 or np.mean(row) > 150:
            break
        cy2 = ty + 1
    return [cx1, cy1, cx2, cy2]


# ---------------------------------------------------------------------------
# Block deduplication helper
# ---------------------------------------------------------------------------

def _expand_text_box_with_ink(
    tbox: List[int],
    container_box: List[int],
    gray: Optional[np.ndarray],
    mask_u8: Optional[np.ndarray],
) -> List[int]:
    """
    Expands a DBNet-derived text_box inside `container_box` by unioning:
      1) Active UNet ink pixels (`mask_u8 > 35`) inside a 4px inset of `container_box`.
      2) For dark-background boxes (`_classify_dark_text_block`), bright text strokes
         (`gray > median(interior) + 45`, filtered by text-sized connected components).
    This prevents missing lines on dark boxes (e.g. 'WHY DID I...') when DBNet only catches 1 line.
    """
    bx1, by1, bx2, by2 = [int(v) for v in container_box]
    tx1, ty1, tx2, ty2 = [int(v) for v in tbox]

    ix1 = min(bx2 - 1, bx1 + 4)
    iy1 = min(by2 - 1, by1 + 4)
    ix2 = max(ix1 + 1, bx2 - 4)
    iy2 = max(iy1 + 1, by2 - 4)
    if ix2 <= ix1 + 2 or iy2 <= iy1 + 2:
        return [tx1, ty1, tx2, ty2]

    # 1. Union with active UNet mask_u8 > 85 inside inset container
    if mask_u8 is not None and mask_u8.size > 0:
        mh, mw = mask_u8.shape[:2]
        my1, my2 = max(0, iy1), min(mh, iy2)
        mx1, mx2 = max(0, ix1), min(mw, ix2)
        if my2 > my1 and mx2 > mx1:
            ys, xs = np.where(mask_u8[my1:my2, mx1:mx2] > 85)
            if xs.size > 0:
                tx1 = min(tx1, mx1 + int(xs.min()))
                ty1 = min(ty1, my1 + int(ys.min()))
                tx2 = max(tx2, mx1 + int(xs.max()) + 1)
                ty2 = max(ty2, my1 + int(ys.max()) + 1)

    # 2. Union with bright text strokes inside dark boxes
    if gray is not None and gray.size > 0 and _classify_dark_text_block(gray, container_box):
        gh, gw = gray.shape[:2]
        gy1, gy2 = max(0, iy1), min(gh, iy2)
        gx1, gx2 = max(0, ix1), min(gw, ix2)
        if gy2 > gy1 and gx2 > gx1:
            interior = gray[gy1:gy2, gx1:gx2]
            if interior.size > 0:
                thresh = float(np.median(interior)) + 45.0
                bright = (interior > thresh).astype(np.uint8)
                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
                int_w = gx2 - gx1
                int_h = gy2 - gy1
                max_comp_area = max(32, int(0.25 * interior.size))
                for lbl_i in range(1, num_labels):
                    cw = int(stats[lbl_i, cv2.CC_STAT_WIDTH])
                    ch = int(stats[lbl_i, cv2.CC_STAT_HEIGHT])
                    carea = int(stats[lbl_i, cv2.CC_STAT_AREA])
                    if carea < 3 or carea > max_comp_area:
                        continue
                    if cw > int(0.75 * int_w) or ch > int(0.45 * int_h):
                        continue
                    cx1 = gx1 + int(stats[lbl_i, cv2.CC_STAT_LEFT])
                    cy1 = gy1 + int(stats[lbl_i, cv2.CC_STAT_TOP])
                    cx2 = cx1 + cw
                    cy2 = cy1 + ch
                    tx1 = min(tx1, cx1)
                    ty1 = min(ty1, cy1)
                    tx2 = max(tx2, cx2)
                    ty2 = max(ty2, cy2)

    tx1 = max(bx1, tx1)
    ty1 = max(by1, ty1)
    tx2 = min(bx2, tx2)
    ty2 = min(by2, ty2)
    if tx2 <= tx1 + 2 or ty2 <= ty1 + 2:
        return [int(v) for v in tbox]
    return [tx1, ty1, tx2, ty2]


def _box_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    a1 = float(max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1]))
    a2 = float(max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _max_contiguous_true(arr: np.ndarray) -> int:
    """Return the length of the longest contiguous True run in a 1D boolean array."""
    if arr.size == 0:
        return 0
    padded = np.concatenate(([False], arr.astype(bool), [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if starts.size == 0:
        return 0
    return int(np.max(ends - starts))


def _has_continuous_dark_vertical_border(strip: np.ndarray, min_span_ratio: float = 0.58) -> bool:
    """
    Return True only if at least one column in `strip` has a continuous dark ink
    stroke spanning >= min_span_ratio of the strip height (ignoring stacked text lines).
    """
    if strip.size == 0 or strip.shape[0] == 0:
        return False
    req_span = max(4, int(round(min_span_ratio * strip.shape[0])))
    dark = strip < 115
    cand_cols = np.where(np.sum(dark, axis=0) >= req_span)[0]
    for c in cand_cols:
        if _max_contiguous_true(dark[:, c]) >= req_span:
            return True
    return False


def _merge_same_bubble_rtdetr_boxes(
    rt_bubbles: List[Dict],
    lines: List[np.ndarray],
    gray: Optional[np.ndarray],
    cc_labels: Optional[np.ndarray] = None,
    mask_u8: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Iteratively merge overlapping RT-DETR bubble boxes that belong to the same single
    smooth speech bubble (high containment, shared horizontal text line, or continuous
    non-pinched white oval) while preserving genuinely touching bubbles separated by
    an ink stroke or a sharp figure-8 concave pinch.
    """
    if len(rt_bubbles) < 2:
        return [dict(b) for b in rt_bubbles]

    # Precompute line bounding boxes for Condition B
    line_bboxes: List[Tuple[int, int, int, int]] = []
    for line in (lines if lines is not None else []):
        l_arr = np.asarray(line, dtype=np.int32)
        if l_arr.size == 0:
            continue
        lx1, ly1 = int(l_arr[:, 0].min()), int(l_arr[:, 1].min())
        lx2, ly2 = int(l_arr[:, 0].max()), int(l_arr[:, 1].max())
        if lx2 > lx1 and ly2 > ly1:
            line_bboxes.append((lx1, ly1, lx2, ly2))

    current = [
        {"box": list(b["box"]), "score": float(b.get("score", 0.0)), "label": b.get("label", "bubble")}
        for b in rt_bubbles
    ]
    im_h = gray.shape[0] if (gray is not None and gray.size > 0) else 0
    im_w = gray.shape[1] if (gray is not None and gray.size > 0) else 0

    changed = True
    while changed:
        changed = False
        current.sort(
            key=lambda b: (
                (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]),
                float(b.get("score", 0.0)),
            ),
            reverse=True,
        )
        next_boxes: List[Dict] = []
        for cand_b in current:
            cb = cand_b["box"]
            ca = max(1, (cb[2] - cb[0]) * (cb[3] - cb[1]))
            cw = max(1, cb[2] - cb[0])
            ch = max(1, cb[3] - cb[1])
            merged = False

            for prev_b in next_boxes:
                pb = prev_b["box"]
                pa = max(1, (pb[2] - pb[0]) * (pb[3] - pb[1]))
                pw = max(1, pb[2] - pb[0])
                ph = max(1, pb[3] - pb[1])

                ix1, iy1 = max(cb[0], pb[0]), max(cb[1], pb[1])
                ix2, iy2 = min(cb[2], pb[2]), min(cb[3], pb[3])
                if ix2 < ix1 - 4 or iy2 <= iy1:
                    continue

                oh = iy2 - iy1
                y_ov = oh / float(min(ch, ph))
                if y_ov < 0.60:
                    continue

                # Check for dark dividing ink border between box centers (middle 60% of overlap height)
                has_border = False
                cx1 = int(round(0.5 * (pb[0] + pb[2])))
                cx2 = int(round(0.5 * (cb[0] + cb[2])))
                cy1 = int(round(0.5 * (pb[1] + pb[3])))
                cy2 = int(round(0.5 * (cb[1] + cb[3])))
                if gray is not None and gray.size > 0:
                    sx1, sx2 = min(cx1, cx2), max(cx1, cx2)
                    if sx2 > sx1 + 2:
                        my1 = max(0, iy1 + int(0.20 * oh))
                        my2 = min(im_h, iy2 - int(0.20 * oh))
                        if my2 <= my1:
                            my1, my2 = max(0, iy1), min(im_h, iy2)
                        strip = gray[my1:my2, max(0, sx1):min(im_w, sx2)]
                        if _has_continuous_dark_vertical_border(strip, min_span_ratio=0.58):
                            has_border = True

                if has_border:
                    continue

                inter = max(0, ix2 - ix1) * oh
                union_a = max(1, ca + pa - inter)
                iou = inter / float(union_a)
                contain = inter / float(min(ca, pa))

                # Anti-merge guard 1: union triggers 2D figure-8 lobe split
                union_box = [min(pb[0], cb[0]), min(pb[1], cb[1]), max(pb[2], cb[2]), max(pb[3], cb[3])]
                if split_joined_bubble_lobes is not None and gray is not None and gray.size > 0:
                    try:
                        if len(split_joined_bubble_lobes(gray, union_box, mask_u8)) >= 2:
                            continue
                    except Exception:
                        pass

                # Anti-merge guard 2: low IoU + offset centers + distinct text lines in each box
                if iou < 0.50 and (abs(cx1 - cx2) >= 0.25 * min(pw, cw) or abs(cy1 - cy2) >= 0.25 * min(ph, ch)) and line_bboxes:
                    has_p_only = False
                    has_c_only = False
                    for lx1, ly1, lx2, ly2 in line_bboxes:
                        lcx, lcy = 0.5 * (lx1 + lx2), 0.5 * (ly1 + ly2)
                        in_p = pb[0] <= lcx <= pb[2] and pb[1] <= lcy <= pb[3]
                        in_c = cb[0] <= lcx <= cb[2] and cb[1] <= lcy <= cb[3]
                        if in_p and not in_c:
                            has_p_only = True
                        elif in_c and not in_p:
                            has_c_only = True
                        if has_p_only and has_c_only:
                            break
                    if has_p_only and has_c_only:
                        continue

                # Condition A: High containment
                cond_a = inter > 0 and contain >= 0.65 and y_ov >= 0.80

                # Condition B: Shared horizontal DBNet text line spanning both boxes (>= 20% line width in each)
                cond_b = False
                if not cond_a and line_bboxes:
                    for lx1, ly1, lx2, ly2 in line_bboxes:
                        lw = max(1, lx2 - lx1)
                        if min(ly2, iy2) <= max(ly1, iy1):
                            continue
                        ov_p = max(0, min(lx2, pb[2]) - max(lx1, pb[0])) / float(lw)
                        ov_c = max(0, min(lx2, cb[2]) - max(lx1, cb[0])) / float(lw)
                        if ov_p >= 0.20 and ov_c >= 0.20:
                            cond_b = True
                            break

                # Condition C: Continuous single oval (no concave figure-8 pinch at overlap midpoint)
                cond_c = False
                if not cond_a and not cond_b and inter > 0 and gray is not None and gray.size > 0:
                    if y_ov >= 0.82 and (ix2 - ix1) >= max(6, int(0.08 * min(pw, cw))):
                        x_mid = max(0, min(im_w - 1, (ix1 + ix2) // 2))
                        gy1 = max(0, iy1)
                        gy2 = min(im_h, iy2)
                        if gy2 > gy1:
                            white_span = _max_contiguous_true(gray[gy1:gy2, x_mid] > 170)
                            if white_span >= 0.68 * float(gy2 - gy1):
                                cond_c = True

                # Condition D: Touching halves inside the same white connected component with no pinch & no ink valley
                cond_d = False
                if (
                    not cond_a
                    and not cond_b
                    and not cond_c
                    and y_ov >= 0.75
                    and ix2 >= ix1 - 4
                    and cc_labels is not None
                    and gray is not None
                    and gray.size > 0
                ):
                    c1y = max(0, min(im_h - 1, cy1))
                    c1x = max(0, min(im_w - 1, cx1))
                    c2y = max(0, min(im_h - 1, cy2))
                    c2x = max(0, min(im_w - 1, cx2))
                    lbl1 = int(cc_labels[c1y, c1x])
                    lbl2 = int(cc_labels[c2y, c2x])
                    if lbl1 > 0 and lbl1 == lbl2:
                        is_pinched = False
                        if has_pinched_bubble_waist is not None:
                            try:
                                is_pinched = bool(has_pinched_bubble_waist(gray, pb, cb))
                            except Exception:
                                is_pinched = False
                        if not is_pinched:
                            # Check that there is no clean vertical white ink valley separating the halves
                            sx1, sx2 = min(c1x, c2x), max(c1x, c2x)
                            has_valley = False
                            if sx2 > sx1 + 6:
                                vy1 = max(0, iy1 + int(0.15 * oh))
                                vy2 = min(im_h, iy2 - int(0.15 * oh))
                                if vy2 > vy1:
                                    ink_cols = np.sum(gray[vy1:vy2, sx1:sx2] < 165, axis=0)
                                    if _max_contiguous_true(ink_cols <= 1) >= max(8, int(0.08 * (sx2 - sx1))):
                                        has_valley = True
                            if not has_valley:
                                cond_d = True

                if cond_a or cond_b or cond_c or cond_d:
                    prev_b["box"] = [
                        min(pb[0], cb[0]),
                        min(pb[1], cb[1]),
                        max(pb[2], cb[2]),
                        max(pb[3], cb[3]),
                    ]
                    prev_b["score"] = max(float(prev_b.get("score", 0.0)), float(cand_b.get("score", 0.0)))
                    merged = True
                    changed = True
                    break

            if not merged:
                next_boxes.append(cand_b)
        current = next_boxes

    return current


def _merge_overlapping_same_bubble_results(
    results: List[Dict],
    gray: Optional[np.ndarray],
) -> List[Dict]:
    """
    Post-assembly guard: merge two speech_bubble results whose outer boxes AND
    inner text_boxes overlap horizontally with no dark vertical border between
    their text_box centers.
    """
    if len(results) < 2:
        return results

    im_h = gray.shape[0] if (gray is not None and gray.size > 0) else 0
    im_w = gray.shape[1] if (gray is not None and gray.size > 0) else 0

    current = [dict(r) for r in results]
    changed = True
    while changed:
        changed = False
        next_res: List[Dict] = []
        for cand in current:
            if cand.get("category") != "speech_bubble":
                next_res.append(cand)
                continue

            cb = cand["box"]
            ctb = cand.get("text_box") or cb
            cw_tb = max(1, ctb[2] - ctb[0])
            ch_tb = max(1, ctb[3] - ctb[1])
            ch_b = max(1, cb[3] - cb[1])

            merged = False
            for prev in next_res:
                if prev.get("category") != "speech_bubble":
                    continue
                pb = prev["box"]
                ptb = prev.get("text_box") or pb
                pw_tb = max(1, ptb[2] - ptb[0])
                ph_tb = max(1, ptb[3] - ptb[1])
                ph_b = max(1, pb[3] - pb[1])

                # Check outer box overlap (y_ov >= 0.70 and x_inter > 0)
                bx_inter = min(cb[2], pb[2]) - max(cb[0], pb[0])
                by_inter = min(cb[3], pb[3]) - max(cb[1], pb[1])
                if bx_inter <= 0 or by_inter <= 0:
                    continue
                if (by_inter / float(min(ch_b, ph_b))) < 0.70:
                    continue

                # Check inner text_box overlap
                tx_inter = min(ctb[2], ptb[2]) - max(ctb[0], ptb[0])
                ty_i1 = max(ctb[1], ptb[1])
                ty_i2 = min(ctb[3], ptb[3])
                ty_inter = ty_i2 - ty_i1
                if tx_inter <= 0 or ty_inter <= 0:
                    continue

                min_tw = float(min(cw_tb, pw_tb))
                min_th = float(min(ch_tb, ph_tb))
                tb_ov = (tx_inter / min_tw >= 0.25 and ty_inter / min_th >= 0.45)
                if not tb_ov:
                    continue

                # Check for dark vertical/curved border between text_box centers across outer bubble height
                has_border = False
                if gray is not None and gray.size > 0:
                    tcx1 = (ptb[0] + ptb[2]) // 2
                    tcx2 = (ctb[0] + ctb[2]) // 2
                    tcy1 = (ptb[1] + ptb[3]) // 2
                    tcy2 = (ctb[1] + ctb[3]) // 2
                    sx1, sx2 = min(tcx1, tcx2), max(tcx1, tcx2)
                    if sx2 > sx1 + 2:
                        by_i1 = max(cb[1], pb[1])
                        by_i2 = min(cb[3], pb[3])
                        boh = by_i2 - by_i1
                        my1 = max(0, by_i1 + int(0.15 * boh))
                        my2 = min(im_h, by_i2 - int(0.15 * boh))
                        if my2 <= my1:
                            my1, my2 = max(0, by_i1), min(im_h, by_i2)
                        strip = gray[my1:my2, max(0, sx1):min(im_w, sx2)]
                        if _has_continuous_dark_vertical_border(strip, min_span_ratio=0.45):
                            has_border = True
                        elif strip.size > 0 and float(np.mean(np.min(strip, axis=1) < 115)) >= 0.55:
                            has_border = True
                    sy1, sy2 = min(tcy1, tcy2), max(tcy1, tcy2)
                    if not has_border and sy2 > sy1 + 2:
                        bx_i1 = max(cb[0], pb[0])
                        bx_i2 = min(cb[2], pb[2])
                        bow = bx_i2 - bx_i1
                        mx1 = max(0, bx_i1 + int(0.15 * bow))
                        mx2 = min(im_w, bx_i2 - int(0.15 * bow))
                        if mx2 <= mx1:
                            mx1, mx2 = max(0, bx_i1), min(im_w, bx_i2)
                        h_strip = gray[max(0, sy1):min(im_h, sy2), mx1:mx2]
                        if h_strip.size > 0 and float(np.mean(np.min(h_strip, axis=0) < 115)) >= 0.55:
                            has_border = True

                if has_border:
                    continue

                prev["box"] = [
                    min(pb[0], cb[0]),
                    min(pb[1], cb[1]),
                    max(pb[2], cb[2]),
                    max(pb[3], cb[3]),
                ]
                prev["text_box"] = [
                    min(ptb[0], ctb[0]),
                    min(ptb[1], ctb[1]),
                    max(ptb[2], ctb[2]),
                    max(ptb[3], ctb[3]),
                ]
                prev["confidence"] = max(float(prev.get("confidence", 0.0)), float(cand.get("confidence", 0.0)))
                merged = True
                changed = True
                break

            if not merged:
                next_res.append(cand)
        current = next_res

    return current


def _deduplicate_blocks(results: List[Dict], iou_thresh: float = 0.40, containment_thresh: float = 0.60) -> List[Dict]:
    """
    Removes duplicate / nested detections overlapping the same speech bubble or text region
    (e.g. a partial DBNet scatter cluster nested inside a merged block) while preserving
    distinct pinched lobes or side-by-side bubbles.
    """
    if len(results) <= 1:
        for i, item in enumerate(results, start=1):
            item["block_id"] = f"b_{i}"
        return results

    def _area(b: List[int]) -> float:
        return float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))

    # Prefer larger, higher-confidence detections first so outer unified blocks win over fragments
    ordered = sorted(
        results,
        key=lambda d: (_area(d.get("text_box") or d["box"]), float(d.get("confidence", 0.0))),
        reverse=True,
    )
    kept: List[Dict] = []
    for cand in ordered:
        tb_c = cand.get("text_box") or cand["box"]
        area_c = _area(tb_c)
        if area_c <= 0:
            continue
        merged_into_existing = False
        for prev in kept:
            tb_p = prev.get("text_box") or prev["box"]
            area_p = _area(tb_p)
            ix1 = max(tb_c[0], tb_p[0])
            iy1 = max(tb_c[1], tb_p[1])
            ix2 = min(tb_c[2], tb_p[2])
            iy2 = min(tb_c[3], tb_p[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = float((ix2 - ix1) * (iy2 - iy1))
            union = area_c + area_p - inter
            iou = inter / union if union > 0 else 0.0
            contain = inter / min(area_c, area_p) if min(area_c, area_p) > 0 else 0.0
            if iou >= iou_thresh or contain >= containment_thresh:
                # Expand text_box union cleanly in case the suppressed candidate caught an extra character
                prev["text_box"] = [
                    min(tb_p[0], tb_c[0]),
                    min(tb_p[1], tb_c[1]),
                    max(tb_p[2], tb_c[2]),
                    max(tb_p[3], tb_c[3]),
                ]
                pb = prev["box"]
                cb = cand["box"]
                if contain >= 0.85:
                    # Only absorb box if nested; avoid merging separate adjacent white contours
                    prev["box"] = [
                        min(pb[0], cb[0]),
                        min(pb[1], cb[1]),
                        max(pb[2], cb[2]),
                        max(pb[3], cb[3]),
                    ]
                prev["confidence"] = max(float(prev.get("confidence", 0.0)), float(cand.get("confidence", 0.0)))
                merged_into_existing = True
                break
        if not merged_into_existing:
            kept.append(cand)

    if _crop_reading_key is not None:
        kept.sort(key=lambda d: _crop_reading_key(d, reading_direction="rtl"))
    else:
        kept.sort(key=lambda d: (d["box"][1] // 120, -d["box"][2]))

    for i, item in enumerate(kept, start=1):
        item["block_id"] = f"b_{i}"
    return kept


# ---------------------------------------------------------------------------
# Public ComicTextDetector singleton / wrapper
# ---------------------------------------------------------------------------

_shared_ctd_detector = None
_shared_ctd_lock = threading.Lock()


def get_shared_comic_detector(model_dir: Optional[str] = None) -> "ComicTextDetector":
    global _shared_ctd_detector
    if _shared_ctd_detector is not None:
        return _shared_ctd_detector
    with _shared_ctd_lock:
        if _shared_ctd_detector is None:
            _shared_ctd_detector = ComicTextDetector(model_dir=model_dir)
    return _shared_ctd_detector


class ComicTextDetector:
    """
    Thread-safe wrapper for mayocream/comic-text-detector (YOLOv5 + UNet + DBNet).
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        input_size: int = 1024,
        conf_thresh: float = 0.38,
        nms_thresh: float = 0.35,
        box_thresh: float = 0.52,
        mask_thresh: float = 0.3,
        half: Optional[bool] = None,
    ):
        self.model_dir = model_dir or os.environ.get("MANGA_CTD_MODEL_DIR", "/opt/models/comic_text_detector")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.box_thresh = box_thresh
        self.mask_thresh = mask_thresh
        if half is None:
            self.half = (self.device == "cuda") and (os.environ.get("MANGA_USE_FP16", "1").lower() in ("1", "true", "yes"))
        else:
            self.half = bool(half and self.device == "cuda")

        self.seg_rep = _SegDetectorRepresenter(thresh=mask_thresh, box_thresh=box_thresh)
        self._net: Optional[TextDetNet] = None
        self._load_failed = False
        self._lock = threading.RLock()

    def _resolve_model_dir(self) -> Optional[Path]:
        candidates = [
            Path(self.model_dir),
            Path("/opt/models/comic_text_detector"),
            Path(__file__).resolve().parent / "checkpoints" / "comic_text_detector",
        ]
        for cand in candidates:
            if cand.is_dir() and (cand / "yolo-v5.safetensors").is_file():
                return cand
        return None

    def load_model(self) -> Optional[TextDetNet]:
        with self._lock:
            if self._net is not None:
                return self._net
            if self._load_failed:
                return None

            local_dir = self._resolve_model_dir()
            if local_dir is None:
                if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
                    logger.warning("ComicTextDetector weights missing at %s under HF_HUB_OFFLINE=1", self.model_dir)
                    self._load_failed = True
                    return None
                try:
                    from huggingface_hub import snapshot_download
                    dl_path = snapshot_download(
                        repo_id="mayocream/comic-text-detector",
                        local_dir=self.model_dir,
                    )
                    local_dir = Path(dl_path)
                except Exception as exc:
                    logger.warning("Could not download mayocream/comic-text-detector: %s", exc)
                    self._load_failed = True
                    return None

            try:
                from safetensors.torch import load_file

                yolo_sd = load_file(str(local_dir / "yolo-v5.safetensors"), device="cpu")
                unet_sd = load_file(str(local_dir / "unet.safetensors"), device="cpu")
                dbnet_sd = load_file(str(local_dir / "dbnet.safetensors"), device="cpu")

                blk_det = YoloV5Backbone(_YOLOV5S_CTD_CFG)
                yolo_sd_f32 = {k: (v.float() if v.is_floating_point() else v) for k, v in yolo_sd.items()}
                blk_det.load_state_dict(yolo_sd_f32, strict=True)
                blk_det.out_indices = [1, 3, 5, 7, 9]

                text_seg = UnetHead(act="leaky")
                text_seg.load_state_dict(unet_sd, strict=True)

                text_det = DBHead(64, act="leaky")
                text_det.load_state_dict(dbnet_sd, strict=True)

                net = TextDetNet(blk_det, text_seg, text_det)
                net.eval()
                net.fuse()
                net.to(self.device)
                if self.half:
                    net.half()
                self._net = net
                logger.info("ComicTextDetector loaded from %s on %s (half=%s)", local_dir, self.device, self.half)
                return self._net
            except Exception as exc:
                logger.error("Failed to initialize ComicTextDetector: %s", exc, exc_info=True)
                self._load_failed = True
                return None

    @property
    def is_available(self) -> bool:
        return self.load_model() is not None

    def detect(self, pil_image: Image.Image) -> Tuple[List[Dict], np.ndarray]:
        """
        Detects comic text blocks (speech bubbles, dark narration frames, borderless text)
        and returns (blocks, binary_unet_mask).
        """
        rgb = np.asarray(pil_image.convert("RGB"))
        im_h, im_w = rgb.shape[:2]
        empty_mask = np.zeros((im_h, im_w), dtype=np.uint8)

        net = self.load_model()
        if net is None:
            return [], empty_mask

        # Preserve exact dmMaze/comic-text-detector channel convention
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        img_rgb_for_lb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_lb, _, (dw, dh) = _letterbox(img_rgb_for_lb, new_shape=self.input_size, stride=64)
        img_chw = np.ascontiguousarray(img_lb.transpose((2, 0, 1))[::-1])
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        img_tensor = img_tensor / 255.0

        with self._lock:
            with torch.inference_mode():
                blks_raw, mask_raw, lines_map = net(img_tensor)

        valid_w = max(1, self.input_size[1] - dw)
        valid_h = max(1, self.input_size[0] - dh)
        resize_ratio = (im_w / float(valid_w), im_h / float(valid_h))

        det = _non_max_suppression(blks_raw, conf_thres=self.conf_thresh, iou_thres=self.nms_thresh, agnostic=True)[0]
        if det.device.type != "cpu":
            det = det.detach().cpu().numpy()
        else:
            det = det.numpy()
        if det.shape[0] > 0:
            det[:, [0, 2]] *= resize_ratio[0]
            det[:, [1, 3]] *= resize_ratio[1]
            blines = det[:, 0:4].astype(np.int32)
            confs = np.round(det[:, 4], 3)
            cls_arr = det[:, 5].astype(np.int32)
        else:
            blines = np.zeros((0, 4), dtype=np.int32)
            confs = np.zeros((0,), dtype=np.float32)
            cls_arr = np.zeros((0,), dtype=np.int32)

        mask_np = mask_raw.squeeze().detach().cpu().float().numpy()
        mask_u8 = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
        mask_u8 = mask_u8[: mask_u8.shape[0] - dh, : mask_u8.shape[1] - dw]
        mask_u8 = cv2.resize(mask_u8, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

        lines_batch, scores_batch = self.seg_rep(lines_map)
        if len(scores_batch) > 0 and len(scores_batch[0]) > 0:
            keep = np.where(scores_batch[0] > self.box_thresh)[0]
            lines = lines_batch[0][keep]
            if lines.size > 0:
                lines = lines.astype(np.float64)
                lines[..., 0] *= resize_ratio[0]
                lines[..., 1] *= resize_ratio[1]
                lines = lines.astype(np.int32)
            else:
                lines = []
        else:
            lines = []

        # Binary UNet text mask for downstream cleaning
        mask_bin = (mask_u8 > int(self.mask_thresh * 255)).astype(np.uint8) * 255
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        cc_labels = None
        cc_stats = None
        if get_white_cc_labels is not None:
            try:
                cc_labels, cc_stats = get_white_cc_labels(gray, return_stats=True)
            except Exception:
                cc_labels, cc_stats = None, None

        # Optional Hybrid Bubble Instance Detection via RT-DETR-v2
        rtdetr_engine = None
        if get_shared_bubble_rtdetr is not None:
            try:
                eng = get_shared_bubble_rtdetr()
                if eng.is_available:
                    rtdetr_engine = eng
            except Exception as e_rt:
                logger.debug("RT-DETR unavailable: %s", e_rt)

        rtdetr_dets = []
        if rtdetr_engine is not None:
            try:
                rtdetr_dets = rtdetr_engine.detect(pil_image)
            except Exception as e_det:
                logger.warning("RT-DETR detect error: %s", e_det)
                rtdetr_dets = []

        rt_bubbles = [d for d in rtdetr_dets if d.get("label") == "bubble"]
        rt_text_bubbles = [d for d in rtdetr_dets if d.get("label") == "text_bubble"]
        rt_text_free = [d for d in rtdetr_dets if d.get("label") == "text_free"]
        raw_candidates: List[Tuple[_CTDBlock, List[int], Optional[List[int]]]] = []

        if rt_bubbles and (len(lines) > 0 or len(rt_text_bubbles) > 0):
            # Iteratively merge redundant overlapping RT-DETR bubble boxes on the same single oval bubble
            rt_bubbles = _merge_same_bubble_rtdetr_boxes(rt_bubbles, lines, gray, cc_labels=cc_labels, mask_u8=mask_u8)

            assigned_lines_per_bubble: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(rt_bubbles))}
            unassigned_lines: List[np.ndarray] = []

            for line in lines:
                l_arr = np.asarray(line, dtype=np.int32)
                lx1, ly1 = int(l_arr[:, 0].min()), int(l_arr[:, 1].min())
                lx2, ly2 = int(l_arr[:, 0].max()), int(l_arr[:, 1].max())
                l_area = max(1, (lx2 - lx1) * (ly2 - ly1))
                lcx, lcy = 0.5 * (lx1 + lx2), 0.5 * (ly1 + ly2)

                best_rt_idx = -1
                best_area = float("inf")
                for rt_idx, rt_b in enumerate(rt_bubbles):
                    rx1, ry1, rx2, ry2 = rt_b["box"]
                    if rx1 <= lcx <= rx2 and ry1 <= lcy <= ry2:
                        area_b = (rx2 - rx1) * (ry2 - ry1)
                        if area_b < best_area:
                            best_area = area_b
                            best_rt_idx = rt_idx

                if best_rt_idx < 0:
                    best_overlap = 0.0
                    for rt_idx, rt_b in enumerate(rt_bubbles):
                        inter_a = _union_area(rt_b["box"], [lx1, ly1, lx2, ly2])
                        overlap = inter_a / float(l_area)
                        if overlap > 0.40 and overlap > best_overlap:
                            best_overlap = overlap
                            best_rt_idx = rt_idx

                if best_rt_idx >= 0:
                    assigned_lines_per_bubble[best_rt_idx].append(line)
                else:
                    unassigned_lines.append(line)

            for rt_idx, rt_b in enumerate(rt_bubbles):
                assigned = assigned_lines_per_bubble[rt_idx]
                if not assigned:
                    rx1, ry1, rx2, ry2 = rt_b["box"]
                    synth_lines = []
                    if rt_text_bubbles:
                        for tb_det in rt_text_bubbles:
                            if float(tb_det.get("score", 0.0)) < 0.35:
                                continue
                            tbx1, tby1, tbx2, tby2 = tb_det["box"]
                            tb_area = max(1, (tbx2 - tbx1) * (tby2 - tby1))
                            ix1, iy1 = max(rx1, tbx1), max(ry1, tby1)
                            ix2, iy2 = min(rx2, tbx2), min(ry2, tby2)
                            if ix2 > ix1 and iy2 > iy1:
                                contain = ((ix2 - ix1) * (iy2 - iy1)) / float(tb_area)
                                if contain >= 0.50:
                                    synth_lines.append(
                                        np.array(
                                            [[tbx1, tby1], [tbx2, tby1], [tbx2, tby2], [tbx1, tby2]],
                                            dtype=np.int32,
                                        )
                                    )
                    if not synth_lines and mask_u8 is not None and mask_u8.size > 0:
                        ix1 = max(0, rx1 + 4)
                        iy1 = max(0, ry1 + 4)
                        ix2 = min(im_w, rx2 - 4)
                        iy2 = min(im_h, ry2 - 4)
                        if ix2 > ix1 and iy2 > iy1:
                            high_texture = False
                            if gray is not None and gray.shape[0] >= iy2 and gray.shape[1] >= ix2:
                                g_sub = gray[iy1:iy2, ix1:ix2]
                                if g_sub.size > 0 and float(np.median(g_sub)) < 190.0 and float(np.std(g_sub)) > 55.0:
                                    high_texture = True
                            if not high_texture:
                                sub_m = mask_u8[iy1:iy2, ix1:ix2] > 90
                                if int(np.sum(sub_m)) >= max(60, int(0.015 * sub_m.size)):
                                    ys_m, xs_m = np.where(sub_m)
                                    mx1, my1 = ix1 + int(xs_m.min()), iy1 + int(ys_m.min())
                                    mx2, my2 = ix1 + int(xs_m.max()) + 1, iy1 + int(ys_m.max()) + 1
                                    if mx2 > mx1 + 2 and my2 > my1 + 2:
                                        synth_lines.append(
                                            np.array(
                                                [[mx1, my1], [mx2, my1], [mx2, my2], [mx1, my2]],
                                                dtype=np.int32,
                                            )
                                        )
                    if synth_lines:
                        assigned = synth_lines
                if not assigned:
                    continue
                orig_b_box = list(rt_b["box"])
                rt_conf = float(rt_b.get("score", 0.85))

                def _append_split_subs(subs_list: List[_CTDBlock], box_bounds: List[int]) -> None:
                    for sb in subs_list:
                        _examine_blk(sb, im_w, sort=True)
                        orig_xyxy = list(sb.xyxy)
                        sb.adjust_bbox(with_bbox=False)
                        if orig_xyxy[2] > orig_xyxy[0] + 4 and orig_xyxy[3] > orig_xyxy[1] + 4:
                            ix1 = max(int(sb.xyxy[0]), int(orig_xyxy[0]))
                            iy1 = max(int(sb.xyxy[1]), int(orig_xyxy[1]))
                            ix2 = min(int(sb.xyxy[2]), int(orig_xyxy[2]))
                            iy2 = min(int(sb.xyxy[3]), int(orig_xyxy[3]))
                            if ix2 > ix1 + 4 and iy2 > iy1 + 4:
                                sb.xyxy = [ix1, iy1, ix2, iy2]
                            else:
                                sb.xyxy = orig_xyxy

                    # Determine whether sub-blocks are stacked vertically or side-by-side horizontally
                    xs_c = [0.5 * (float(b.xyxy[0]) + float(b.xyxy[2])) for b in subs_list]
                    ys_c = [0.5 * (float(b.xyxy[1]) + float(b.xyxy[3])) for b in subs_list]
                    is_vertical_stack = (max(ys_c) - min(ys_c)) > (max(xs_c) - min(xs_c))

                    if is_vertical_stack:
                        subs_list.sort(key=lambda b: float(b.xyxy[1]))
                    else:
                        subs_list.sort(key=lambda b: float(b.xyxy[0]))

                    n_subs = len(subs_list)
                    for sub_i, sb in enumerate(subs_list):
                        stx1 = max(0, min(im_w, int(sb.xyxy[0])))
                        sty1 = max(0, min(im_h, int(sb.xyxy[1])))
                        stx2 = max(0, min(im_w, int(sb.xyxy[2])))
                        sty2 = max(0, min(im_h, int(sb.xyxy[3])))
                        if stx2 <= stx1 + 4 or sty2 <= sty1 + 4:
                            continue
                        if is_vertical_stack:
                            top_edge = box_bounds[1] if sub_i == 0 else max(
                                box_bounds[1], min(sty1 - 8, (int(subs_list[sub_i - 1].xyxy[3]) + sty1) // 2)
                            )
                            bot_edge = box_bounds[3] if sub_i == n_subs - 1 else min(
                                box_bounds[3], max(sty2 + 8, (sty2 + int(subs_list[sub_i + 1].xyxy[1])) // 2)
                            )
                            c_box = [box_bounds[0], top_edge, box_bounds[2], bot_edge]
                        else:
                            left_edge = box_bounds[0] if sub_i == 0 else max(
                                box_bounds[0], min(stx1 - 8, (int(subs_list[sub_i - 1].xyxy[2]) + stx1) // 2)
                            )
                            right_edge = box_bounds[2] if sub_i == n_subs - 1 else min(
                                box_bounds[2], max(stx2 + 8, (stx2 + int(subs_list[sub_i + 1].xyxy[0])) // 2)
                            )
                            c_box = [left_edge, box_bounds[1], right_edge, box_bounds[3]]
                        expanded_tb = _expand_text_box_with_ink([stx1, sty1, stx2, sty2], c_box, gray, mask_u8)
                        raw_candidates.append((sb, expanded_tb, c_box))

                # 1. Universal 2D Bubble-Lobe Splitting (Distance Transform + Watershed)
                lobe_handled = False
                if split_joined_bubble_lobes is not None and gray is not None and gray.size > 0:
                    lobes = split_joined_bubble_lobes(gray, orig_b_box, mask_u8)
                    if len(lobes) >= 2:
                        ox1 = max(0, min(im_w, int(orig_b_box[0])))
                        oy1 = max(0, min(im_h, int(orig_b_box[1])))
                        lobe_assigned: List[List[np.ndarray]] = [[] for _ in range(len(lobes))]
                        for line in assigned:
                            l_arr = np.asarray(line, dtype=np.int32)
                            lx1, ly1 = int(l_arr[:, 0].min()), int(l_arr[:, 1].min())
                            lx2, ly2 = int(l_arr[:, 0].max()), int(l_arr[:, 1].max())
                            lcx = int(round(0.5 * (lx1 + lx2))) - ox1
                            lcy = int(round(0.5 * (ly1 + ly2))) - oy1
                            best_l = -1
                            for l_idx, (l_mask, _) in enumerate(lobes):
                                lh, lw = l_mask.shape[:2]
                                if 0 <= lcy < lh and 0 <= lcx < lw and l_mask[lcy, lcx]:
                                    best_l = l_idx
                                    break
                            if best_l < 0:
                                best_ov = 0
                                for l_idx, (l_mask, _) in enumerate(lobes):
                                    lh, lw = l_mask.shape[:2]
                                    px1, py1 = max(0, lx1 - ox1), max(0, ly1 - oy1)
                                    px2, py2 = min(lw, lx2 - ox1), min(lh, ly2 - oy1)
                                    if px2 > px1 and py2 > py1:
                                        ov = int(np.sum(l_mask[py1:py2, px1:px2]))
                                        if ov > best_ov:
                                            best_ov = ov
                                            best_l = l_idx
                            if best_l >= 0:
                                lobe_assigned[best_l].append(line)

                        populated_lobes: List[Tuple[List[np.ndarray], List[int]]] = []
                        for (l_mask, l_box), l_lines in zip(lobes, lobe_assigned):
                            clamped_lbox = [
                                max(orig_b_box[0], l_box[0]),
                                max(orig_b_box[1], l_box[1]),
                                min(orig_b_box[2], l_box[2]),
                                min(orig_b_box[3], l_box[3]),
                            ]
                            if l_lines:
                                populated_lobes.append((l_lines, clamped_lbox))
                            elif mask_u8 is not None and mask_u8.size > 0:
                                lh, lw = l_mask.shape[:2]
                                my2 = min(im_h, oy1 + lh)
                                mx2 = min(im_w, ox1 + lw)
                                if my2 > oy1 and mx2 > ox1:
                                    sub_m = (mask_u8[oy1:my2, ox1:mx2] > 35) & l_mask[: my2 - oy1, : mx2 - ox1]
                                    if int(np.sum(sub_m)) >= 25:
                                        ys_m, xs_m = np.where(sub_m)
                                        sx1, sy1 = ox1 + int(xs_m.min()), oy1 + int(ys_m.min())
                                        sx2, sy2 = ox1 + int(xs_m.max()) + 1, oy1 + int(ys_m.max()) + 1
                                        if sx2 > sx1 + 2 and sy2 > sy1 + 2:
                                            synth_l = np.array([[sx1, sy1], [sx2, sy1], [sx2, sy2], [sx1, sy2]], dtype=np.int32)
                                            populated_lobes.append(([synth_l], clamped_lbox))

                        if len(populated_lobes) >= 2:
                            lobe_handled = True
                            for l_lines, l_box in populated_lobes:
                                l_blk = _CTDBlock(l_box, lines=l_lines, conf=rt_conf)
                                _examine_blk(l_blk, im_w, sort=True)
                                sub_split, sub_list = _split_horizontal_blk_columns(
                                    l_blk, gray, mask_u8, depth=1, bubble_box=l_box
                                )
                                if sub_split and len(sub_list) >= 2:
                                    _append_split_subs(sub_list, l_box)
                                else:
                                    l_blk.adjust_bbox(with_bbox=False)
                                    stx1 = max(0, min(im_w, int(l_blk.xyxy[0])))
                                    sty1 = max(0, min(im_h, int(l_blk.xyxy[1])))
                                    stx2 = max(0, min(im_w, int(l_blk.xyxy[2])))
                                    sty2 = max(0, min(im_h, int(l_blk.xyxy[3])))
                                    if stx2 > stx1 + 4 and sty2 > sty1 + 4:
                                        expanded_tb = _expand_text_box_with_ink([stx1, sty1, stx2, sty2], l_box, gray, mask_u8)
                                        raw_candidates.append((l_blk, expanded_tb, l_box))

                if lobe_handled:
                    continue

                # 2. Fallback: 2D Line Graph + Stepped-Box (X/Y) + Projection Valley Splitting
                b_blk = _CTDBlock(orig_b_box, lines=assigned, conf=rt_conf)
                _examine_blk(b_blk, im_w, sort=True)
                splitted, sub_blks = _split_horizontal_blk_columns(b_blk, gray, mask_u8, bubble_box=orig_b_box)

                if not splitted:
                    b_blk.adjust_bbox(with_bbox=False)
                    stx1 = max(0, min(im_w, int(b_blk.xyxy[0])))
                    sty1 = max(0, min(im_h, int(b_blk.xyxy[1])))
                    stx2 = max(0, min(im_w, int(b_blk.xyxy[2])))
                    sty2 = max(0, min(im_h, int(b_blk.xyxy[3])))
                    expanded_tb = _expand_text_box_with_ink([stx1, sty1, stx2, sty2], orig_b_box, gray, mask_u8)
                    ew = expanded_tb[2] - expanded_tb[0]
                    eh = expanded_tb[3] - expanded_tb[1]
                    is_pinched_exp = False
                    if ew >= 80 and ew >= 0.80 * max(1, eh) and has_pinched_bubble_waist is not None and gray is not None:
                        try:
                            mid_ex = (expanded_tb[0] + expanded_tb[2]) // 2
                            is_pinched_exp = bool(
                                has_pinched_bubble_waist(
                                    gray,
                                    [expanded_tb[0], expanded_tb[1], mid_ex, expanded_tb[3]],
                                    [mid_ex, expanded_tb[1], expanded_tb[2], expanded_tb[3]],
                                )
                            )
                        except Exception:
                            is_pinched_exp = False
                    if (ew >= 90 and ew >= 1.15 * max(1, eh)) or is_pinched_exp:
                        exp_poly = np.array(
                            [
                                [expanded_tb[0], expanded_tb[1]],
                                [expanded_tb[2], expanded_tb[1]],
                                [expanded_tb[2], expanded_tb[3]],
                                [expanded_tb[0], expanded_tb[3]],
                            ],
                            dtype=np.int32,
                        )
                        tmp_blk = _CTDBlock(orig_b_box, lines=[exp_poly], conf=rt_conf)
                        splitted, sub_blks = _split_horizontal_blk_columns(tmp_blk, gray, mask_u8, bubble_box=orig_b_box)

                if splitted and len(sub_blks) >= 2:
                    _append_split_subs(sub_blks, orig_b_box)
                else:
                    b_blk.adjust_bbox(with_bbox=False)
                    stx1 = max(0, min(im_w, int(b_blk.xyxy[0])))
                    sty1 = max(0, min(im_h, int(b_blk.xyxy[1])))
                    stx2 = max(0, min(im_w, int(b_blk.xyxy[2])))
                    sty2 = max(0, min(im_h, int(b_blk.xyxy[3])))
                    if stx2 > stx1 + 4 and sty2 > sty1 + 4:
                        expanded_tb = _expand_text_box_with_ink([stx1, sty1, stx2, sty2], orig_b_box, gray, mask_u8)
                        raw_candidates.append((b_blk, expanded_tb, orig_b_box))

            # Process unassigned lines and YOLO blocks as fallback
            unassigned_blk_list = _group_output((blines, cls_arr, confs), unassigned_lines, im_w, im_h, mask_u8, gray=gray)
            for blk in unassigned_blk_list:
                tx1 = max(0, min(im_w, int(blk.xyxy[0])))
                ty1 = max(0, min(im_h, int(blk.xyxy[1])))
                tx2 = max(0, min(im_w, int(blk.xyxy[2])))
                ty2 = max(0, min(im_h, int(blk.xyxy[3])))
                if tx2 - tx1 < 6 or ty2 - ty1 < 6:
                    continue
                tbox = [tx1, ty1, tx2, ty2]
                t_area = max(1, (tx2 - tx1) * (ty2 - ty1))
                already_covered = False
                covered_inter = 0
                for _, prev_tb, _ in raw_candidates:
                    if _box_iou(tbox, prev_tb) >= 0.35:
                        already_covered = True
                        break
                    ix1 = max(tx1, prev_tb[0])
                    iy1 = max(ty1, prev_tb[1])
                    ix2 = min(tx2, prev_tb[2])
                    iy2 = min(ty2, prev_tb[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter_a = (ix2 - ix1) * (iy2 - iy1)
                        prev_a = max(1, (prev_tb[2] - prev_tb[0]) * (prev_tb[3] - prev_tb[1]))
                        if inter_a / float(min(t_area, prev_a)) >= 0.50:
                            already_covered = True
                            break
                        covered_inter += inter_a
                if not already_covered and (covered_inter / float(t_area)) >= 0.45:
                    already_covered = True
                if not already_covered:
                    raw_candidates.append((blk, tbox, None))
        else:
            blk_list = _group_output((blines, cls_arr, confs), lines, im_w, im_h, mask_u8, gray=gray)
            for blk in blk_list:
                tx1 = max(0, min(im_w, int(blk.xyxy[0])))
                ty1 = max(0, min(im_h, int(blk.xyxy[1])))
                tx2 = max(0, min(im_w, int(blk.xyxy[2])))
                ty2 = max(0, min(im_h, int(blk.xyxy[3])))
                if tx2 - tx1 >= 6 and ty2 - ty1 >= 6:
                    raw_candidates.append((blk, [tx1, ty1, tx2, ty2], None))

        all_tboxes = [tb for _, tb, _ in raw_candidates]
        results: List[Dict] = []
        for idx, (blk, tbox, box_override) in enumerate(raw_candidates):
            tx1, ty1, tx2, ty2 = tbox
            other_tboxes = [b for j, b in enumerate(all_tboxes) if j != idx]
            t_patch = gray[ty1:ty2, tx1:tx2]
            p75_t = float(np.percentile(t_patch, 75)) if t_patch.size > 0 else 255.0

            is_dark = _classify_dark_text_block(gray, tbox)

            if is_dark:
                box = _expand_dark_narration_box(gray, tbox, other_tboxes)
                bw_b = max(1, box[2] - box[0])
                bh_b = max(1, box[3] - box[1])
                box_patch = gray[box[1]:box[3], box[0]:box[2]]
                dark_ratio = float(np.mean(box_patch < 130)) if box_patch.size > 0 else 0.0
                category = "narration" if dark_ratio >= 0.45 and (bw_b >= 20 and bh_b >= 20) else "speech_bubble"
            elif box_override is not None:
                category = "speech_bubble"
                box = list(box_override)
                # Refine RT-DETR box against other text boxes to avoid crossing neighbors
                for ot in other_tboxes:
                    ox1, oy1, ox2, oy2 = ot
                    v_ov = min(ty2, oy2) - max(ty1, oy1)
                    v_req = 0.20 * max(1, min(ty2 - ty1, oy2 - oy1))
                    if ox2 <= tx1 and v_ov >= v_req:
                        box[0] = max(box[0], (ox2 + tx1) // 2)
                    elif ox1 >= tx2 and v_ov >= v_req:
                        box[2] = min(box[2], (tx2 + ox1) // 2)
                    h_ov = min(tx2, ox2) - max(tx1, ox1)
                    h_req = 0.20 * max(1, min(tx2 - tx1, ox2 - ox1))
                    if oy2 <= ty1 and h_ov >= h_req:
                        box[1] = max(box[1], (oy2 + ty1) // 2)
                    elif oy1 >= ty2 and h_ov >= h_req:
                        box[3] = min(box[3], (ty2 + oy1) // 2)
            else:
                # Check RT-DETR text_free vs text_bubble before white-CC / light-uniform check (avoid SFX -> speech_bubble)
                t_area = max(1, (tx2 - tx1) * (ty2 - ty1))
                is_rt_text_free = False
                if rt_text_free:
                    matches_tf = False
                    for tf_det in rt_text_free:
                        tfb = tf_det["box"]
                        tf_area = max(1, (tfb[2] - tfb[0]) * (tfb[3] - tfb[1]))
                        inter_tf = _union_area(tbox, tfb)
                        if inter_tf > 0:
                            iou_tf = inter_tf / float(t_area + tf_area - inter_tf)
                            contain_tf = inter_tf / float(min(t_area, tf_area))
                            if iou_tf >= 0.30 or contain_tf >= 0.50:
                                matches_tf = True
                                break
                    if matches_tf:
                        max_tb_contain = 0.0
                        for tb_det in rt_text_bubbles:
                            tbb = tb_det["box"]
                            tb_area = max(1, (tbb[2] - tbb[0]) * (tbb[3] - tbb[1]))
                            inter_tb = _union_area(tbox, tbb)
                            if inter_tb > 0:
                                max_tb_contain = max(max_tb_contain, inter_tb / float(min(t_area, tb_area)))
                        if max_tb_contain < 0.30:
                            is_rt_text_free = True

                if is_rt_text_free:
                    category = "free_text"
                    box = [max(0, tx1 - 6), max(0, ty1 - 6), min(im_w, tx2 + 6), min(im_h, ty2 + 6)]
                else:
                    # Check if inside a clean speech bubble vs borderless floating text
                    in_white_cc = False
                    if cc_labels is not None and cc_stats is not None:
                        cx = min(im_w - 1, max(0, (tx1 + tx2) // 2))
                        cy = min(im_h - 1, max(0, (ty1 + ty2) // 2))
                        lbl = int(cc_labels[cy, cx])
                        if lbl > 0:
                            cc_area = int(cc_stats[lbl, cv2.CC_STAT_AREA])
                            if 100 <= cc_area <= 0.25 * im_w * im_h:
                                in_white_cc = True

                    light_bg_est = t_patch[t_patch >= 140] if t_patch.size > 0 else np.array([])
                    is_light_uniform = (
                        p75_t >= 180.0
                        and (light_bg_est.size >= max(8, int(t_patch.size * 0.35)))
                        and (float(np.std(light_bg_est)) < 26.0 if light_bg_est.size > 0 else False)
                    )
                    if in_white_cc or is_light_uniform:
                        category = "speech_bubble"
                        if expand_to_bubble_contour is not None:
                            box = expand_to_bubble_contour(gray, tbox, other_tboxes)
                        else:
                            box = [max(0, tx1 - 10), max(0, ty1 - 10), min(im_w, tx2 + 10), min(im_h, ty2 + 10)]
                    else:
                        category = "free_text"
                        box = [max(0, tx1 - 6), max(0, ty1 - 6), min(im_w, tx2 + 6), min(im_h, ty2 + 6)]

            results.append({
                "block_id": f"b_{len(results) + 1}",
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "text_box": [tx1, ty1, tx2, ty2],
                "category": category,
                "is_dark": bool(is_dark),
                "confidence": float(blk.conf),
                "vertical": bool(blk.vertical),
            })

        results = _merge_overlapping_same_bubble_results(results, gray)
        return _deduplicate_blocks(results), mask_bin
