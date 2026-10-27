"""Forecast windows at 1x, 2x and 3x. Frame numbers are 1-based."""

from __future__ import annotations

N_FRAMES = 20
N_IN = 5
N_OUT = 5

# Speed-augmented training (models C and D): five inputs and at least two targets.
SPEED_STARTS: dict[int, tuple[int, ...]] = {
    1: tuple(range(1, 15)),  # t = 1..14
    2: tuple(range(1, 9)),   # t = 1..8
    3: (1, 2),
}
# Normal-speed training (models A and B).
NORMAL_STARTS: tuple[int, ...] = tuple(range(1, 12))  # t = 1..11


def window_frames(t: int, k: int, n_frames: int = N_FRAMES) -> tuple[list[int], list[int]]:
    """Input and target frame numbers for a window starting at frame t and speed k.

    Targets past frame `n_frames` are dropped. Inputs must all exist.
    """
    inputs = [t + i * k for i in range(N_IN)]
    if any(f < 1 or f > n_frames for f in inputs):
        raise ValueError(f"input frames {inputs} are not inside 1..{n_frames}")
    targets = [t + (N_IN + i) * k for i in range(N_OUT)]
    targets = [f for f in targets if f <= n_frames]
    return inputs, targets


def allowed_starts(k: int, speed_augmented: bool) -> tuple[int, ...]:
    if not speed_augmented:
        if k != 1:
            raise ValueError("normal-speed training uses k = 1 only")
        return NORMAL_STARTS
    try:
        return SPEED_STARTS[k]
    except KeyError as exc:
        raise ValueError(f"speed k must be 1, 2 or 3, got {k}") from exc
