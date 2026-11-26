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
    from detection import expand_to_bubble_contour, get_white_cc_labels
except (ImportError, ValueError):
    try:
        from .detection import expand_to_bubble_contour, get_white_cc_labels
    except (ImportError, ValueError):
        expand_to_bubble_contour = None
        get_white_cc_labels = None

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


def _group_output(blks, lines, im_w: int, im_h: int, mask: Optional[np.ndarray]) -> List[_CTDBlock]:
    blk_list: List[_CTDBlock] = []
    scattered_lines = {"ver": [], "hor": []}
    for bbox, cls, conf in zip(*blks):
        lang = LANG_LIST[int(cls)] if 0 <= int(cls) < len(LANG_LIST) else "unknown"
        blk_list.append(_CTDBlock(bbox, language=lang, conf=float(conf)))

    bbox_score_thresh = 0.4
    mask_score_thresh = 0.1
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
        if len(blk.lines) > 1 and (blk.language == "ja" or blk.vertical):
            splitted, sub_blk_list = _split_blk_on_gap(blk)
        else:
            sub_blk_list = [blk]
        if not splitted:
            for sb in sub_blk_list:
                sb.adjust_bbox(with_bbox=True)
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
            cx1 = tx
            break
        cx1 = tx
    for d in range(1, max_pad):
        tx = x2 + d
        if tx > lim_r:
            break
        col = gray[y1:y2, tx]
        if col.size == 0 or np.mean(col > 170) >= 0.40 or np.mean(col) > 150:
            cx2 = tx
            break
        cx2 = tx
    for d in range(1, max_pad):
        ty = y1 - d
        if ty < lim_t:
            break
        row = gray[ty, cx1:cx2]
        if row.size == 0 or np.mean(row > 170) >= 0.40 or np.mean(row) > 150:
            cy1 = ty
            break
        cy1 = ty
    for d in range(1, max_pad):
        ty = y2 + d
        if ty > lim_b:
            break
        row = gray[ty, cx1:cx2]
        if row.size == 0 or np.mean(row > 170) >= 0.40 or np.mean(row) > 150:
            cy2 = ty
            break
        cy2 = ty
    return [cx1, cy1, cx2, cy2]


# ---------------------------------------------------------------------------
# Block deduplication helper
# ---------------------------------------------------------------------------

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
        conf_thresh: float = 0.32,
        nms_thresh: float = 0.35,
        box_thresh: float = 0.48,
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

        valid_w = max(1, self.input_size[0] - dw)
        valid_h = max(1, self.input_size[1] - dh)
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

        blk_list = _group_output((blines, cls_arr, confs), lines, im_w, im_h, mask_u8)

        # Binary UNet text mask for downstream cleaning
        mask_bin = (mask_u8 > int(self.mask_thresh * 255)).astype(np.uint8) * 255

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        raw_tboxes = []
        for blk in blk_list:
            tx1 = max(0, min(im_w, int(blk.xyxy[0])))
            ty1 = max(0, min(im_h, int(blk.xyxy[1])))
            tx2 = max(0, min(im_w, int(blk.xyxy[2])))
            ty2 = max(0, min(im_h, int(blk.xyxy[3])))
            if tx2 - tx1 >= 6 and ty2 - ty1 >= 6:
                raw_tboxes.append((blk, [tx1, ty1, tx2, ty2]))

        cc_labels = None
        cc_stats = None
        if get_white_cc_labels is not None:
            try:
                cc_labels, cc_stats = get_white_cc_labels(gray, return_stats=True)
            except Exception:
                cc_labels, cc_stats = None, None

        all_tboxes = [tb for _, tb in raw_tboxes]
        results: List[Dict] = []
        for idx, (blk, tbox) in enumerate(raw_tboxes):
            tx1, ty1, tx2, ty2 = tbox
            other_tboxes = [b for j, b in enumerate(all_tboxes) if j != idx]
            t_patch = gray[ty1:ty2, tx1:tx2]
            p75_t = float(np.percentile(t_patch, 75)) if t_patch.size > 0 else 255.0

            is_dark = _classify_dark_text_block(gray, tbox)

            if is_dark:
                box = _expand_dark_narration_box(gray, tbox, other_tboxes)
                # Distinguish rectangular narration frames from dark speech bubbles
                bw_b = max(1, box[2] - box[0])
                bh_b = max(1, box[3] - box[1])
                box_patch = gray[box[1]:box[3], box[0]:box[2]]
                dark_ratio = float(np.mean(box_patch < 130)) if box_patch.size > 0 else 0.0
                category = "narration" if dark_ratio >= 0.45 and (bw_b >= 20 and bh_b >= 20) else "speech_bubble"
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

        return _deduplicate_blocks(results), mask_bin
