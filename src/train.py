from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FER2013Dataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models import create_model
from src.utils.checkpoints import save_checkpoint
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed


def run_training(
    csv_path,
    model_name: str,
    output_dir,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 0,
    fast_dev_run: bool = False,
    model_kwargs: dict | None = None,
    use_scheduler: bool = True,
) -> dict:
    """
    Train a model on FER-2013 data and save the best checkpoint by val macro-F1.

    Returns:
        {
            "best_checkpoint": Path,
            "best_val_macro_f1": float,
            "history": list[dict],
        }
    """
    set_seed(seed)

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{model_name}.pt"

    train_dataset = FER2013Dataset(csv_path, split="train", transform=get_train_transforms())
    val_dataset = FER2013Dataset(csv_path, split="val", transform=get_eval_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, **(model_kwargs or {}))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate / 50) if use_scheduler and not fast_dev_run else None

    # Print run header (suppressed in fast_dev_run)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    if not fast_dev_run:
        print(f"\n{'=' * 60}")
        print(f"  Model   : {model_name}")
        print(f"  Device  : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
        print(f"  Params  : {trainable:,} trainable / {total:,} total")
        print(f"  Train   : {len(train_dataset):,} samples  ({len(train_loader)} batches)")
        print(f"  Val     : {len(val_dataset):,} samples  ({len(val_loader)} batches)")
        print(f"  Epochs  : {epochs}  |  Batch: {batch_size}  |  LR: {learning_rate}")
        print(f"{'=' * 60}\n")

    best_val_macro_f1: float = -1.0
    history: list[dict] = []

    # Outer epoch bar — disabled in fast_dev_run to keep test output clean
    epoch_bar = tqdm(
        range(epochs),
        desc="Epochs",
        unit="ep",
        disable=fast_dev_run,
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"  train {epoch + 1:>3}/{epochs}",
            unit="batch",
            leave=False,
            disable=fast_dev_run,
            dynamic_ncols=True,
        )
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

            train_bar.set_postfix(
                loss=f"{train_loss_sum / (train_bar.n or 1):.4f}",
                acc=f"{train_correct / max(train_total, 1):.3f}",
            )

            if fast_dev_run:
                break

        train_loss = train_loss_sum / max(len(train_bar), 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Val ────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        if len(val_dataset) > 0:
            val_bar = tqdm(
                val_loader,
                desc=f"  val   {epoch + 1:>3}/{epochs}",
                unit="batch",
                leave=False,
                disable=fast_dev_run,
                dynamic_ncols=True,
            )
            with torch.no_grad():
                for images, labels in val_bar:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss_sum += loss.item()
                    val_batches += 1

                    preds = outputs.argmax(dim=1)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())

                    val_bar.set_postfix(loss=f"{val_loss_sum / val_batches:.4f}")

                    if fast_dev_run:
                        break

        val_loss = val_loss_sum / max(val_batches, 1)

        if y_true:
            val_metrics = compute_classification_metrics(y_true, y_pred)
            val_macro_f1: float = float(val_metrics["macro_f1"])
            val_accuracy: float = float(val_metrics["accuracy"])
            val_weighted_f1: float = float(val_metrics["weighted_f1"])
        else:
            val_macro_f1 = 0.0
            val_accuracy = 0.0
            val_weighted_f1 = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
        }
        history.append(epoch_record)

        if scheduler is not None:
            scheduler.step()

        is_best = val_macro_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_macro_f1
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_record,
                model_name=model_name,
            )

        # Update outer bar with current epoch metrics
        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            tr_acc=f"{train_acc:.3f}",
            vl_loss=f"{val_loss:.4f}",
            vl_f1=f"{val_macro_f1:.3f}",
            best=f"{best_val_macro_f1:.3f}" + (" *" if is_best else ""),
        )

    if not fast_dev_run:
        import json

        history_path = output_dir / f"history_{model_name}.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        from src.utils.plotting import save_training_curves

        curves_path = output_dir / "figures" / f"curves_{model_name}.png"
        save_training_curves(history, curves_path)

        print(f"\nDone. Best val macro-F1: {best_val_macro_f1:.4f}")
        print(f"Checkpoint : {checkpoint_path}")
        print(f"History    : {history_path}")
        print(f"Curves     : {curves_path}\n")

    return {
        "best_checkpoint": checkpoint_path,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default="baseline_cnn")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        default=False,
        help="Disable cosine LR scheduler (default: scheduler enabled).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze ResNet-18 backbone (default: True). Use --no-freeze-backbone to train all layers.",
    )
    args = parser.parse_args()

    model_kwargs = {}
    if args.model == "resnet18":
        model_kwargs["freeze_backbone"] = args.freeze_backbone

    result = run_training(
        csv_path=args.csv,
        model_name=args.model,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        model_kwargs=model_kwargs,
        use_scheduler=not args.no_scheduler,
    )
