import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion matrix heatmap to path as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_training_curves(history: list[dict], path) -> None:
    """Save train/val loss and val macro-F1 curves to path as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()

    axes[1].plot(epochs, val_macro_f1, label="val_macro_f1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation macro-F1")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
