"""Mirror, rotation, and the eight symmetries of the square."""

from __future__ import annotations

import torch

# (mirror, quarter-turns anticlockwise). Apply rotation first, then mirror,
# matching (M R^a)(x) := M(R^a(x)). R^a is its own family; each M R^a is an involution.
GROUP: tuple[tuple[bool, int], ...] = tuple(
    [(False, a) for a in range(4)] + [(True, a) for a in range(4)]
)


def apply_group(x: torch.Tensor, mirror: bool, turns: int) -> torch.Tensor:
    """Rotate 90° anticlockwise `turns` times, then optionally mirror left-right."""
    turns = int(turns) % 4
    if turns:
        x = torch.rot90(x, k=turns, dims=(-2, -1))
    if mirror:
        x = torch.flip(x, dims=(-1,))
    return x


def group_inverse(mirror: bool, turns: int) -> tuple[bool, int]:
    turns = int(turns) % 4
    if mirror:
        return True, turns
    return False, (-turns) % 4


def mirror(x: torch.Tensor) -> torch.Tensor:
    return apply_group(x, True, 0)


def rotate(x: torch.Tensor, turns: int = 1) -> torch.Tensor:
    return apply_group(x, False, turns)
