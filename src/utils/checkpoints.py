from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch: int, metrics: dict, model_name: str) -> None:
    """Save model state, optimizer state, epoch, metrics, and model_name to path."""
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, model, map_location="cpu") -> dict:
    """Load checkpoint from path into model. Returns the checkpoint dict (for metrics etc)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
