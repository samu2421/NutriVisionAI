"""
Export the validation accuracy stored in the training checkpoints to JSON.

Training (src/train.py) saved a checkpoint every 5 epochs plus the best model, but
no per-epoch history. This script collects the validation accuracy stored in those
files so the notebook can plot it without needing the (large, untracked) .pth files.

Usage (from the project root):
    python scripts/export_checkpoint_metrics.py

Reads:  models/checkpoints/checkpoint_epoch_*.pth, models/efficientnet_b1_indian.pth
Writes: results/checkpoint_val_accuracy.json
"""

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
BEST_MODEL = ROOT / "models" / "efficientnet_b1_indian.pth"
OUTPUT = ROOT / "results" / "checkpoint_val_accuracy.json"


def read_checkpoint(path: Path, is_best: bool) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": int(ckpt["epoch"]) + 1,  # stored 0-based in src/train.py
        "val_accuracy": float(ckpt["val_acc"]),  # percent, as stored
        "file": str(path.relative_to(ROOT)),
        "saved_best_model": is_best,
    }


def main():
    paths = sorted(CHECKPOINT_DIR.glob("checkpoint_epoch_*.pth"))
    if not paths or not BEST_MODEL.exists():
        raise SystemExit("Checkpoint files not found; run src/train.py first.")

    points = [read_checkpoint(p, False) for p in paths]
    points.append(read_checkpoint(BEST_MODEL, True))
    points.sort(key=lambda p: p["epoch"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "description": "Validation accuracy (%) stored in the saved training checkpoints",
        "checkpoints": points,
    }, indent=2) + "\n")

    for p in points:
        flag = "  <- saved best model" if p["saved_best_model"] else ""
        print(f"epoch {p['epoch']:>2}: {p['val_accuracy']:.2f}%{flag}")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
