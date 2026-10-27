"""ConvLSTM forecaster. Models A–C share weights layout; D dilates layer 2."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Model A, counted from the equations in the handout.
MODEL_A_PARAMS = 480_784
MODEL_D_PARAM_LIMIT = 721_176  # 1.5 × model A

# Folded grid: pixel_unshuffle(4) makes each cell 4 original pixels across.
# The patch width itself is not added to the reach.
CELL_PIXELS = 4


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden: int, dilation: int = 1):
        super().__init__()
        self.hidden = hidden
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_channels + hidden,
            4 * hidden,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )

    def forward(
        self, u: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # channels_last is the faster oneDNN layout on CPU. On CUDA the
        # weights stay NCHW, and converting every step only adds copies.
        if u.device.type == "cpu":
            fmt = torch.channels_last
            u = u.contiguous(memory_format=fmt)
            h = h.contiguous(memory_format=fmt)
            c = c.contiguous(memory_format=fmt)
        i, f, o, g = self.conv(torch.cat([u, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Forecaster(nn.Module):
    """Nine-step rollout. Steps 1–4 are discarded; steps 5–9 are the five predictions.

    From step 6 onward the input is the network's own previous prediction.
    """

    def __init__(self, hidden: int = 64, dilations: tuple[int, int] = (1, 1)):
        super().__init__()
        self.hidden = hidden
        self.dilations = dilations
        self.cell1 = ConvLSTMCell(16, hidden, dilations[0])
        self.cell2 = ConvLSTMCell(hidden, hidden, dilations[1])
        self.head = nn.Conv2d(hidden, 16, kernel_size=1, bias=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, 5, 1, 64, 64) -> predictions (B, 5, 1, 64, 64)
        batch = frames.shape[0]
        h1 = frames.new_zeros(batch, self.hidden, 16, 16)
        c1 = frames.new_zeros(batch, self.hidden, 16, 16)
        h2 = frames.new_zeros(batch, self.hidden, 16, 16)
        c2 = frames.new_zeros(batch, self.hidden, 16, 16)
        preds: list[torch.Tensor] = []
        prev: torch.Tensor | None = None
        for step in range(9):
            frame = frames[:, step] if step < 5 else prev
            u = F.pixel_unshuffle(frame, 4)
            h1, c1 = self.cell1(u, h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
            prev = torch.sigmoid(F.pixel_shuffle(self.head(h2), 4))
            if step >= 4:
                preds.append(prev)
        return torch.stack(preds, dim=1)


def build_model(name: str) -> Forecaster:
    """A, B and C are the baseline. D uses a dilated second ConvLSTM."""
    key = name.upper().replace("+", "")
    if key in {"A", "B", "C"}:
        return Forecaster(dilations=(1, 1))
    if key in {"D", "DB"}:
        # Layer 1 stays a dense 3×3 (radius 1 cell) so adjacent cells are visible.
        # Layer 2 uses dilation 3 (radius 3). The summed offsets fill every cell
        # out to Chebyshev radius 4, with no holes, and the parameter count is
        # unchanged. Reach is 16 original pixels, against model A's 8. Figure 1
        # puts 3× motion near 10 px/frame, with a 90th percentile near 12.
        return Forecaster(dilations=(1, 3))
    raise ValueError(f"unknown model {name}")


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reach_cells(dilations: tuple[int, int]) -> int:
    """Chebyshev radius, in folded-grid cells, of one rollout step.

    A 3×3 kernel with dilation d reads offsets {-d, 0, +d}. Stacked layers add
    those radii. The 1×1 head, pixel_unshuffle and pixel_shuffle do not move
    information. The longest path into the prediction is the new input (or
    layer 1's previous state) through both convolutions.
    """
    return sum(dilations)


def reach_pixels(dilations: tuple[int, int]) -> int:
    """Axis-aligned centre-to-centre reach in original pixels."""
    return CELL_PIXELS * reach_cells(dilations)
