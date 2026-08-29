# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
# from mmcv.runner import  auto_fp16


class Grid(object):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def _build_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = int(torch.randint(2, h, ()).item())
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        st_h = int(torch.randint(d, ()).item())
        st_w = int(torch.randint(d, ()).item())

        row_mask = None
        col_mask = None
        if self.use_h:
            rows = torch.arange(hh, device=device)
            row_mask = (rows >= st_h) & (((rows - st_h) % d) < self.l)
        if self.use_w:
            cols = torch.arange(ww, device=device)
            col_mask = (cols >= st_w) & (((cols - st_w) % d) < self.l)

        if row_mask is not None and col_mask is not None:
            mask = ~(row_mask[:, None] | col_mask[None, :])
        elif row_mask is not None:
            mask = ~row_mask[:, None].expand(hh, ww)
        elif col_mask is not None:
            mask = ~col_mask[None, :].expand(hh, ww)
        else:
            mask = torch.ones((hh, ww), dtype=torch.bool, device=device)

        return mask.to(dtype=torch.float32)

    def _rotate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.rotate <= 1:
            return mask

        angle = int(torch.randint(self.rotate, ()).item())
        if angle == 0:
            return mask

        radians = math.radians(angle)
        theta = mask.new_tensor(
            [[[math.cos(radians), -math.sin(radians), 0.0], [math.sin(radians), math.cos(radians), 0.0]]]
        )
        mask = mask[None, None]
        grid = F.affine_grid(theta, mask.shape, align_corners=False)
        return F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]

   # @auto_fp16()
    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.prob:
            return x
        _, _, h, w = x.size()
        mask = self._build_mask(h, w, x.device)
        mask = self._rotate_mask(mask)
        hh, ww = mask.shape
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.to(dtype=x.dtype).view(1, 1, h, w)
        if self.offset:
            offset = 2 * (torch.rand((1, 1, h, w), dtype=x.dtype, device=x.device) - 0.5)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x
