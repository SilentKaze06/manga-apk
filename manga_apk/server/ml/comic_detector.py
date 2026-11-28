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
    from detection import deduplicate_text_detections, match_bubbles_to_text_blocks
except (ImportError, ValueError):
    try:
        from .detection import deduplicate_text_detections, match_bubbles_to_text_blocks
    except (ImportError, ValueError):
        try:
            from ml.detection import deduplicate_text_detections, match_bubbles_to_text_blocks
        except (ImportError, ValueError):
            deduplicate_text_detections = None
            match_bubbles_to_text_blocks = None

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


# ---------------------------------------------------------------------------
# Public ComicTextDetector singleton / wrapper
# ---------------------------------------------------------------------------

def _deduplicate_blocks(results: List[Dict], iou_thresh: float = 0.40, containment_thresh: float = 0.60) -> List[Dict]:
    kept = [dict(d) for d in deduplicate_text_detections(results, iou_thresh=iou_thresh, contain_thresh=containment_thresh)]
    for i, item in enumerate(kept, start=1):
        item["block_id"] = f"b_{i}"
    return kept


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
        3-stage neural detection:
        1. RT-DETR-v2 (`ogkalu/comic-text-and-bubble-detector`) detects unified text blocks
           (`text_bubble`, `text_free`) and speech bubble bounds (`bubble` metadata only).
        2. Light class-agnostic NMS / containment deduplication (zero pixel merge/split).
        3. CTD UNet (`mayocream/comic-text-detector`) produces pixel-level character mask,
           strictly intersected with the union of RT-DETR-v2 text boxes so non-text artwork
           is never masked.
        """
        rgb = np.asarray(pil_image.convert("RGB"))
        im_h, im_w = rgb.shape[:2]
        empty_mask = np.zeros((im_h, im_w), dtype=np.uint8)

        # Stage 1: RT-DETR-v2 block & bubble detection
        rtdetr_engine = None
        if get_shared_bubble_rtdetr is not None:
            try:
                eng = get_shared_bubble_rtdetr()
                if eng.is_available:
                    rtdetr_engine = eng
            except Exception as e_rt:
                logger.debug("RT-DETR unavailable: %s", e_rt)

        rtdetr_dets: List[Dict] = []
        if rtdetr_engine is not None:
            try:
                rtdetr_dets = rtdetr_engine.detect(pil_image)
            except Exception as e_det:
                logger.warning("RT-DETR detect error: %s", e_det)
                rtdetr_dets = []

        rt_bubbles = [d for d in rtdetr_dets if d.get("label") == "bubble"]
        rt_text = [d for d in rtdetr_dets if d.get("label") in ("text_bubble", "text_free")]

        # Light class-agnostic NMS / containment deduplication (no pixel-distance splitting)
        if deduplicate_text_detections is not None:
            rt_text = deduplicate_text_detections(rt_text, iou_thresh=0.50, contain_thresh=0.75)

        if not rt_text:
            return [], empty_mask

        if match_bubbles_to_text_blocks is not None:
            rt_text = match_bubbles_to_text_blocks(rt_text, rt_bubbles)

        # Stage 2: CTD UNet pixel-level character segmentation mask (ignoring CTD box merging)
        ctd_mask_bin = empty_mask
        net = self.load_model()
        if net is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            img_rgb_for_lb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_lb, _, (dw, dh) = _letterbox(img_rgb_for_lb, new_shape=self.input_size, stride=64)
            img_chw = np.ascontiguousarray(img_lb.transpose((2, 0, 1))[::-1])
            img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
            img_tensor = img_tensor.half() if self.half else img_tensor.float()
            img_tensor = img_tensor / 255.0

            with self._lock:
                with torch.inference_mode():
                    _, mask_raw, _ = net(img_tensor)

            mask_np = mask_raw.squeeze().detach().cpu().float().numpy()
            mask_u8 = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
            mask_u8 = mask_u8[: mask_u8.shape[0] - dh, : mask_u8.shape[1] - dw]
            mask_u8 = cv2.resize(mask_u8, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            ctd_mask_bin = (mask_u8 > int(self.mask_thresh * 255)).astype(np.uint8) * 255

        # Intersect CTD mask with union of RT-DETR-v2 text boxes so non-text artwork is never masked
        text_union_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        results: List[Dict] = []
        for det in rt_text:
            tb = det["box"]
            tx1 = max(0, min(im_w, int(tb[0])))
            ty1 = max(0, min(im_h, int(tb[1])))
            tx2 = max(0, min(im_w, int(tb[2])))
            ty2 = max(0, min(im_h, int(tb[3])))
            if tx2 <= tx1 + 4 or ty2 <= ty1 + 4:
                continue

            text_union_mask[ty1:ty2, tx1:tx2] = 255

            label = det.get("label", "text_bubble")
            category = "speech_bubble" if label == "text_bubble" else "free_text"
            layout_box = det.get("layout_box")
            box = [int(v) for v in layout_box] if layout_box else [tx1, ty1, tx2, ty2]
            is_dark = _classify_dark_text_block(gray, [tx1, ty1, tx2, ty2])

            rec = {
                "block_id": f"b_{len(results) + 1}",
                "box": box,
                "text_box": [tx1, ty1, tx2, ty2],
                "label": label,
                "category": category,
                "is_dark": bool(is_dark),
                "confidence": float(det.get("score", 0.90)),
                "vertical": bool((ty2 - ty1) > (tx2 - tx1) * 1.15),
            }
            if det.get("bubble_box"):
                rec["bubble_box"] = [int(v) for v in det["bubble_box"]]
            results.append(rec)

        page_mask = cv2.bitwise_and(ctd_mask_bin, text_union_mask)
        return results, page_mask
