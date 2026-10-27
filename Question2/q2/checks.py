"""Table 1 and the implementation checks that later results depend on."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import GROUP, apply_group, group_inverse, mirror, rotate
from .metrics import masked_mse
from .model import (
    MODEL_A_PARAMS,
    MODEL_D_PARAM_LIMIT,
    Forecaster,
    build_model,
    reach_pixels,
    trainable_parameters,
)
from .windows import window_frames


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def transformation_checks() -> list[tuple[str, str, str, bool]]:
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1

    rows: list[tuple[str, str, str, bool]] = []

    diff_a = _max_abs(mirror(mirror(x)), x)
    rows.append(("(a)", "0", _fmt_diff(diff_a), diff_a == 0 and torch.equal(mirror(mirror(x)), x)))

    r4 = x
    for _ in range(4):
        r4 = rotate(r4, 1)
    diff_b = _max_abs(r4, x)
    rows.append(("(b)", "0", _fmt_diff(diff_b), diff_b == 0 and torch.equal(r4, x)))

    mrm = mirror(rotate(mirror(x), 1))
    rinv = rotate(x, -1)
    diff_c = _max_abs(mrm, rinv)
    rows.append(("(c)", "0", _fmt_diff(diff_c), diff_c == 0 and torch.equal(mrm, rinv)))

    rows.append(("(d)", "(7, 10, 43)", str(_bright(mirror(p))), _bright(mirror(p)) == (7, 10, 43)))
    rows.append(("(e)", "(7, 43, 10)", str(_bright(rotate(p, 1))), _bright(rotate(p, 1)) == (7, 43, 10)))

    video = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).clone()
    for label, k, expected_in, expected_tg in (
        ("(f)", 1, list(range(1, 6)), list(range(6, 11))),
        ("(g)", 2, [1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        ("(h)", 3, [1, 4, 7, 10, 13], [16, 19]),
    ):
        inputs, targets = window_frames(1, k)
        got_in = [int(video[f - 1, 0, 0, 0]) for f in inputs]
        got_tg = [int(video[f - 1, 0, 0, 0]) for f in targets]
        expected = f"{_frames(expected_in)} / {_frames(expected_tg)}"
        got = f"{_frames(got_in)} / {_frames(got_tg)}"
        rows.append((label, expected, got, got_in == expected_in and got_tg == expected_tg))
    return rows


def _frames(frames: list[int]) -> str:
    if frames == list(range(frames[0], frames[-1] + 1)):
        return f"{frames[0]}–{frames[-1]}"
    return ", ".join(str(f) for f in frames)


def _fmt_diff(value: float) -> str:
    return "0" if value == 0 else f"{value:.3g}"


def _bright(video: torch.Tensor) -> tuple[int, int, int]:
    idx = (video == 1).nonzero(as_tuple=False)
    if idx.shape[0] != 1:
        raise RuntimeError(f"expected one bright pixel, found {idx.shape[0]}")
    frame, _channel, row, col = (int(v) for v in idx[0])
    return frame, row, col


def group_checks() -> None:
    torch.manual_seed(1)
    x = torch.rand(2, 5, 1, 64, 64)
    seen = []
    for mirror_flag, turns in GROUP:
        y = apply_group(x, mirror_flag, turns)
        inv_m, inv_t = group_inverse(mirror_flag, turns)
        back = apply_group(y, inv_m, inv_t)
        if not torch.equal(back, x):
            raise AssertionError(f"inverse failed for mirror={mirror_flag}, turns={turns}")
        seen.append(y.reshape(-1).tolist())
    # The eight symmetries must actually differ.
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            if seen[i] == seen[j]:
                raise AssertionError("two group elements acted identically")


def model_checks() -> None:
    model_a = build_model("A")
    n_a = trainable_parameters(model_a)
    if n_a != MODEL_A_PARAMS:
        raise AssertionError(f"model A has {n_a} parameters, expected {MODEL_A_PARAMS}")
    for name in ("B", "C"):
        if trainable_parameters(build_model(name)) != MODEL_A_PARAMS:
            raise AssertionError(f"model {name} changed the architecture")
    model_d = build_model("D")
    n_d = trainable_parameters(model_d)
    if n_d > MODEL_D_PARAM_LIMIT:
        raise AssertionError(f"model D has {n_d} parameters, limit is {MODEL_D_PARAM_LIMIT}")
    if reach_pixels(model_d.dilations) <= reach_pixels(model_a.dilations):
        raise AssertionError("model D reach is not strictly larger than model A")

    _check_rollout(model_a)
    _check_loss_mask()


def _check_rollout(model: Forecaster) -> None:
    torch.manual_seed(0)
    frames = torch.rand(1, 5, 1, 64, 64)
    seen: list[torch.Tensor] = []
    original = model.cell1.forward

    def record(u, h, c):
        seen.append(u.detach().clone())
        return original(u, h, c)

    model.cell1.forward = record
    pred = model(frames)
    model.cell1.forward = original
    if len(seen) != 9:
        raise AssertionError(f"rollout took {len(seen)} steps, expected 9")
    if pred.shape != (1, 5, 1, 64, 64):
        raise AssertionError(f"prediction shape {tuple(pred.shape)}")
    for step in range(5):
        expect = F.pixel_unshuffle(frames[:, step], 4)
        if not torch.equal(seen[step], expect):
            raise AssertionError(f"step {step + 1} did not read input frame {step + 1}")
    for step, horizon in ((5, 0), (6, 1), (7, 2), (8, 3)):
        expect = F.pixel_unshuffle(pred[:, horizon], 4)
        if not torch.allclose(seen[step], expect):
            raise AssertionError(f"step {step + 1} was not fed prediction horizon {horizon + 1}")


def _check_loss_mask() -> None:
    pred = torch.zeros(2, 5, 1, 8, 8)
    target = torch.zeros_like(pred)
    target[0, 0] = 1
    mask = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, True, True, True],
        ]
    )
    loss = masked_mse(pred, target, mask)
    if not torch.allclose(loss, torch.tensor(0.5)):
        raise AssertionError(f"masked loss should be 0.5, got {float(loss)}")
    pred_invalid = pred.clone()
    pred_invalid[0, 1:] = 5
    if not torch.allclose(masked_mse(pred_invalid, target, mask), loss):
        raise AssertionError("invalid target frames leaked into the loss")


def main() -> None:
    rows = transformation_checks()
    print(f"{'Check':<8}{'Expected':<28}{'Yours':<28}OK")
    failed = False
    for label, expected, got, ok in rows:
        print(f"{label:<8}{expected:<28}{got:<28}{ok}")
        failed = failed or not ok
    group_checks()
    model_checks()
    print(f"model A parameters: {trainable_parameters(build_model('A'))}")
    print(
        f"model A reach: {reach_pixels(model_a.dilations)} px   "
        f"model D reach: {reach_pixels(model_d.dilations)} px"
    )
    if failed:
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
