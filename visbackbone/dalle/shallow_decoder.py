import attr
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections  import OrderedDict
from functools    import partial
from .decoder import DecoderBlock

@attr.s(eq=False)
class Conv2d(nn.Module):
    n_in:  int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1)
    kw:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 2 == 1)


    def __attrs_post_init__(self) -> None:
        super().__init__()

        w = torch.empty((self.n_out, self.n_in, self.kw, self.kw))
        w.normal_(std=1 / math.sqrt(self.n_in * self.kw ** 2))

        b = torch.zeros((self.n_out,))
        self.w, self.b = nn.Parameter(w), nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.w, self.b

        return F.conv2d(x, w, b, padding=(self.kw - 1) // 2)


@attr.s(eq=False, repr=False)
class DecoderBlock(nn.Module):
    n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        make_conv     = partial(Conv2d)
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', make_conv(self.n_in,  self.n_hid, 1)),
                ('relu_2', nn.ReLU()),
                ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_3', nn.ReLU()),
                ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_4', nn.ReLU()),
                ('conv_4', make_conv(self.n_hid, self.n_out, 3)),]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(eq=False, repr=False)
class ShallowVqDecoder(nn.Module):
    group_count:     int = attr.ib(default=1,  validator=lambda i, a, x: x >= 1)
    n_init:          int = attr.ib(default=128,  validator=lambda i, a, x: x >= 8)
    n_hid:           int = attr.ib(default=256,  validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2,    validator=lambda i, a, x: x >= 1)
    vocab_size:      int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        blk_range  = range(self.n_blk_per_group)
        n_layers   = self.group_count * self.n_blk_per_group
        make_conv  = partial(Conv2d)
        make_blk   = partial(DecoderBlock, n_layers=n_layers)

        self.blocks = [
            ('input', make_conv(self.vocab_size, self.n_init, 1))]
        for ng in range(self.group_count):
            if ng == 0:
                self.blocks.append(
                    (f'group_{ng + 1}', nn.Sequential(OrderedDict([
                    *[(f'block_{i + 1}', make_blk(self.n_init if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
                ])))
                )
            elif ng == 1:
                self.blocks.append(
                    (f'group_{ng + 1}', nn.Sequential(OrderedDict([
                    *[(f'block_{i + 1}', make_blk(8 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
                ])))
                )
            elif ng == 2:
                self.blocks.append(
                    (f'group_{ng + 1}', nn.Sequential(OrderedDict([
                    *[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
                ])))
                )
            elif ng == 3:
                self.blocks.append(
                    (f'group_{ng + 1}', nn.Sequential(OrderedDict([
                    *[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
                ])))
                )
        self.blocks = nn.Sequential(OrderedDict(self.blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.vocab_size:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.vocab_size}')

        return self.blocks(x)
