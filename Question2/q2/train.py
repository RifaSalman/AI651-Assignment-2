"""Train one forecaster. The protocol is the one in the handout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import ROOT, TRAIN_IDS, collate_windows, load_videos, sample_window_specs
from .metrics import masked_mse
from .model import build_model

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"

# What changes relative to model A. Architecture changes only for D.
PROTOCOL = {
    "A": {"speed_augmented": False, "symmetry": False},
    "B": {"speed_augmented": False, "symmetry": True},
    "C": {"speed_augmented": True, "symmetry": False},
    "D": {"speed_augmented": True, "symmetry": False},
    "DB": {"speed_augmented": True, "symmetry": True},
}


def checkpoint_path(name: str, seed: int) -> Path:
    return CHECKPOINT_DIR / f"{name}_seed{seed}.pt"


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(name)


def train(
    name: str,
    seed: int,
    epochs: int = 30,
    batch_size: int = 32,
    limit_batches: int = 0,
    resume: bool = True,
    device: torch.device | None = None,
    output_dir: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    protocol = PROTOCOL[name]
    videos = load_videos(data_root)
    train_ids = torch.tensor(list(TRAIN_IDS), dtype=torch.long)
    device = device or resolve_device()
    out = output_dir or ROOT
    log_dir = out / "outputs" / "logs"
    path = out / "checkpoints" / f"{name}_seed{seed}.pt"

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        model = build_model(name).to(device)
        hardware = torch.cuda.get_device_name(device)
    else:
        torch.set_num_threads(4)
        model = build_model(name).to(memory_format=torch.channels_last)
        hardware = "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator()
    generator.manual_seed(seed)

    epoch_seconds: list[float] = []
    epoch_loss: list[float] = []
    start_epoch = 0
    if resume and path.exists() and limit_batches == 0:
        saved = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        generator_state = saved["generator"]
        if torch.is_tensor(generator_state):
            generator_state = generator_state.detach().cpu()
        generator.set_state(generator_state)
        epoch_seconds = list(saved["epoch_seconds"])
        epoch_loss = list(saved["epoch_loss"])
        start_epoch = int(saved["epoch"])
        print(f"resuming {name} seed {seed} from epoch {start_epoch}", flush=True)
    if start_epoch >= epochs:
        print(f"{name} seed {seed} already has {start_epoch} epochs", flush=True)
        return path
    print(
        f"training {name} seed {seed} on {device} ({hardware}), "
        f"epochs {start_epoch + 1}–{epochs}",
        flush=True,
    )

    n_videos = train_ids.shape[0]
    # A limited batch count is a timing check, not a training run.
    if limit_batches:
        epochs = min(epochs, start_epoch + 1)
    for epoch in range(start_epoch, epochs):
        model.train()
        order = train_ids[torch.randperm(n_videos, generator=generator)]
        total = 0.0
        seen = 0
        t0 = time.perf_counter()
        n_batches = n_videos // batch_size
        if limit_batches:
            n_batches = min(n_batches, limit_batches)
        for batch_index in range(n_batches):
            idx = order[batch_index * batch_size : (batch_index + 1) * batch_size]
            t, k, group = sample_window_specs(
                batch_size,
                generator,
                protocol["speed_augmented"],
                protocol["symmetry"],
            )
            inputs, targets, mask = collate_windows(videos, idx, t, k, group)
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(inputs), targets, mask)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            seen += 1
        elapsed = time.perf_counter() - t0
        mean_loss = total / max(seen, 1)
        print(
            f"{name} seed {seed}  epoch {epoch + 1}/{epochs}  "
            f"loss {mean_loss:.4f}  {elapsed:.1f}s",
            flush=True,
        )
        if limit_batches:
            continue
        epoch_seconds.append(elapsed)
        epoch_loss.append(mean_loss)
        _save(
            path,
            log_dir,
            name,
            seed,
            epoch + 1,
            model,
            optimizer,
            generator,
            epoch_seconds,
            epoch_loss,
            hardware,
        )
    if limit_batches:
        return path
    mean_epoch = sum(epoch_seconds) / len(epoch_seconds)
    print(f"mean epoch time: {mean_epoch:.1f}s", flush=True)
    return path


def _save(
    path,
    log_dir,
    name,
    seed,
    epoch,
    model,
    optimizer,
    generator,
    epoch_seconds,
    epoch_loss,
    hardware,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": name,
            "seed": seed,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "epoch_seconds": epoch_seconds,
            "epoch_loss": epoch_loss,
            "dilations": model.dilations,
            "hardware": hardware,
        },
        path,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "model": name,
        "seed": seed,
        "epochs_done": epoch,
        "epoch_seconds": epoch_seconds,
        "epoch_loss": epoch_loss,
        "mean_epoch_seconds": sum(epoch_seconds) / len(epoch_seconds),
        "hardware": hardware,
    }
    (log_dir / f"{name}_seed{seed}.json").write_text(json.dumps(log, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(PROTOCOL))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    train(
        args.model,
        args.seed,
        epochs=args.epochs,
        limit_batches=args.limit_batches,
        resume=not args.no_resume,
        device=resolve_device(args.device),
        output_dir=args.output_dir,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
