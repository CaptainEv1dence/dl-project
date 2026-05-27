from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalization(mean: float, std: float, imagenet_norm: bool):
    if imagenet_norm:
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return transforms.Normalize(mean=[mean], std=[std])


def get_train_transforms(
    mean: float = 0.5,
    std: float = 0.5,
    image_size: int = 48,
    strong_aug: bool = False,
    imagenet_norm: bool = False,
):
    """Augmenting transforms for the training split.

    Defaults keep the old 1-channel 48x48 pipeline.
    For ImageNet-pretrained models use image_size > 48 and imagenet_norm=True.
    """

    ops = [transforms.ToPILImage()]

    if image_size != 48:
        ops.append(transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC))

    if imagenet_norm:
        # Convert grayscale FER images to 3-channel PIL images before ImageNet normalization.
        ops.append(transforms.Grayscale(num_output_channels=3))

    ops.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10)),
        ]
    )

    if strong_aug:
        # Hue/saturation are intentionally omitted because FER-2013 is grayscale.
        ops.append(transforms.ColorJitter(brightness=0.25, contrast=0.25))

    ops.append(transforms.ToTensor())

    if strong_aug:
        # Tensor transform; keep before normalization.
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"))

    ops.append(_normalization(mean, std, imagenet_norm))
    return transforms.Compose(ops)


def get_eval_transforms(
    mean: float = 0.5,
    std: float = 0.5,
    image_size: int = 48,
    imagenet_norm: bool = False,
):
    """Deterministic transforms for validation / test splits."""

    ops = [transforms.ToPILImage()]

    if image_size != 48:
        ops.append(transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC))

    if imagenet_norm:
        ops.append(transforms.Grayscale(num_output_channels=3))

    ops.extend(
        [
            transforms.ToTensor(),
            _normalization(mean, std, imagenet_norm),
        ]
    )
    return transforms.Compose(ops)
