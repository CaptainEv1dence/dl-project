from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import NUM_CLASSES
from src.data.dataset import FER2013Dataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models import create_model
from src.utils.checkpoints import save_checkpoint
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed


def _parse_weights_arg(value: str | None):
    if value is None:
        return None
    if str(value).lower() in {"none", "null", "false", "0"}:
        return None
    return value


def _make_class_weights(dataset: FER2013Dataset, mode: str, device: torch.device) -> torch.Tensor | None:
    """Return class weights for CrossEntropyLoss.

    mode:
    - none: old behavior
    - balanced: total / (num_classes * class_count)
    - sqrt: sqrt of balanced weights, usually less aggressive for FER-2013
    """
    if mode == "none":
        return None

    labels = torch.tensor([label for _, label in dataset._samples], dtype=torch.long)
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float().clamp_min(1.0)
    weights = counts.sum() / (NUM_CLASSES * counts)

    if mode == "sqrt":
        weights = torch.sqrt(weights)
    elif mode != "balanced":
        raise ValueError(f"Unknown class weight mode: {mode!r}")

    # Keep the mean weight near 1 so the loss scale stays stable.
    weights = weights / weights.mean()
    return weights.to(device)


def _count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _set_all_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


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
    label_smoothing: float = 0.0,
    class_weights: str = "none",
    strong_aug: bool = False,
    image_size: int = 48,
    imagenet_norm: bool = False,
    optimizer_name: str = "adamw",
    amp: bool = False,
    grad_clip: float = 0.0,
    unfreeze_epoch: int | None = None,
) -> dict:
    """Train a model on FER-2013 data and save the best checkpoint by val macro-F1."""

    set_seed(seed)

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{model_name}.pt"

    train_dataset = FER2013Dataset(
        csv_path,
        split="train",
        transform=get_train_transforms(
            image_size=image_size,
            strong_aug=strong_aug,
            imagenet_norm=imagenet_norm,
        ),
    )
    val_dataset = FER2013Dataset(
        csv_path,
        split="val",
        transform=get_eval_transforms(
            image_size=image_size,
            imagenet_norm=imagenet_norm,
        ),
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, **(model_kwargs or {}))
    model.to(device)

    weight_tensor = _make_class_weights(train_dataset, class_weights, device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    def make_optimizer_and_scheduler(start_epoch: int = 0):
        params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name!r}")

        remaining = max(epochs - start_epoch, 1)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining,
                eta_min=max(learning_rate / 100, 1e-7),
            )
            if use_scheduler and not fast_dev_run
            else None
        )
        return optimizer, scheduler

    optimizer, scheduler = make_optimizer_and_scheduler(start_epoch=0)
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    trainable, total = _count_trainable(model)
    if not fast_dev_run:
        print(f"\n{'=' * 60}")
        print(f" Model          : {model_name}")
        print(f" Device         : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
        print(f" Params         : {trainable:,} trainable / {total:,} total")
        print(f" Train          : {len(train_dataset):,} samples ({len(train_loader)} batches)")
        print(f" Val            : {len(val_dataset):,} samples ({len(val_loader)} batches)")
        print(f" Epochs         : {epochs} | Batch: {batch_size} | LR: {learning_rate}")
        print(f" Image size     : {image_size} | ImageNet norm: {imagenet_norm}")
        print(f" Loss           : CE(label_smoothing={label_smoothing}, class_weights={class_weights})")
        print(f" Optimizer      : {optimizer_name} | WD: {weight_decay} | AMP: {amp_enabled}")
        print(f" Strong aug     : {strong_aug} | Grad clip: {grad_clip}")
        if unfreeze_epoch is not None:
            print(f" Unfreeze epoch : {unfreeze_epoch}")
        if weight_tensor is not None:
            print(f" Class weights  : {[round(float(x), 3) for x in weight_tensor.detach().cpu()]}")
        print(f"{'=' * 60}\n")

    best_val_macro_f1: float = -1.0
    history: list[dict] = []

    epoch_bar = tqdm(
        range(epochs),
        desc="Epochs",
        unit="ep",
        disable=fast_dev_run,
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            _set_all_trainable(model)
            optimizer, scheduler = make_optimizer_and_scheduler(start_epoch=epoch)
            trainable, total = _count_trainable(model)
            if not fast_dev_run:
                print(f"\nUnfroze full backbone at epoch {epoch}. Trainable params: {trainable:,}/{total:,}\n")

        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            train_loader,
            desc=f" train {epoch + 1:>3}/{epochs}",
            unit="batch",
            leave=False,
            disable=fast_dev_run,
            dynamic_ncols=True,
        )

        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()

            batch_size_now = labels.size(0)
            train_loss_sum += loss.item() * batch_size_now
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += batch_size_now

            train_bar.set_postfix(
                loss=f"{train_loss_sum / max(train_total, 1):.4f}",
                acc=f"{train_correct / max(train_total, 1):.3f}",
            )

            if fast_dev_run:
                break

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Val ────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        if len(val_dataset) > 0:
            val_bar = tqdm(
                val_loader,
                desc=f" val {epoch + 1:>3}/{epochs}",
                unit="batch",
                leave=False,
                disable=fast_dev_run,
                dynamic_ncols=True,
            )

            with torch.no_grad():
                for images, labels in val_bar:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    batch_size_now = labels.size(0)
                    val_loss_sum += loss.item() * batch_size_now
                    val_total += batch_size_now

                    preds = outputs.argmax(dim=1)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())

                    val_bar.set_postfix(loss=f"{val_loss_sum / max(val_total, 1):.4f}")

                    if fast_dev_run:
                        break

        val_loss = val_loss_sum / max(val_total, 1)

        if y_true:
            val_metrics = compute_classification_metrics(y_true, y_pred, labels=list(range(NUM_CLASSES)))
            val_macro_f1 = float(val_metrics["macro_f1"])
            val_accuracy = float(val_metrics["accuracy"])
            val_weighted_f1 = float(val_metrics["weighted_f1"])
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
    parser.add_argument("--no-scheduler", action="store_true", default=False)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weights", default="default", help='Transfer weights: "default" or "none".')
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weights", choices=["none", "balanced", "sqrt"], default="none")
    parser.add_argument("--strong-aug", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--imagenet-norm", action="store_true", default=False)

    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=None,
        help="If set, make all parameters trainable at this 0-based epoch and restart optimizer/scheduler.",
    )

    args = parser.parse_args()

    model_kwargs = {}
    if args.model in {"resnet18", "efficientnet_b2"}:
        model_kwargs["freeze_backbone"] = args.freeze_backbone
        model_kwargs["weights"] = _parse_weights_arg(args.weights)

    if args.model == "efficientnet_b2":
        model_kwargs["dropout"] = args.dropout

    run_training(
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
        label_smoothing=args.label_smoothing,
        class_weights=args.class_weights,
        strong_aug=args.strong_aug,
        image_size=args.image_size,
        imagenet_norm=args.imagenet_norm,
        optimizer_name=args.optimizer,
        amp=args.amp,
        grad_clip=args.grad_clip,
        unfreeze_epoch=args.unfreeze_epoch,
    )
