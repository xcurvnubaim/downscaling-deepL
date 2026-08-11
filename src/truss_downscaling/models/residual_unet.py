from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple(h: int, w: int, multiple: int) -> tuple[int, int]:
    return (-h) % multiple, (-w) % multiple


class ResidualBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if ch_in == ch_out else nn.Conv2d(ch_in, ch_out, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResidualUNet(nn.Module):
    """Depth-4/base-width-64 model used by the production checkpoint."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, depth: int = 4, base_width: int = 64) -> None:
        super().__init__()
        self.depth = depth
        widths = [base_width * 2**i for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.enc_blocks.append(ResidualBlock(previous, width))
            self.pools.append(nn.MaxPool2d(2))
            previous = width
        self.bottleneck = ResidualBlock(previous, base_width * 2**depth)
        self.dec_ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        previous = base_width * 2**depth
        for width in reversed(widths):
            self.dec_ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.dec_blocks.append(ResidualBlock(previous + width, width))
            previous = width
        self.head = nn.Conv2d(previous, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        multiple = 2**self.depth
        ph, pw = pad_to_multiple(x.shape[-2], x.shape[-1], multiple)
        x = F.pad(x, (0, pw, 0, ph))
        skips = []
        for block, pool in zip(self.enc_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.dec_ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat([x, skip], dim=1))
        x = self.head(x)
        x = x[..., : original.shape[-2], : original.shape[-1]]
        if x.shape[1] == original.shape[1]:
            x = x + original
        return x
