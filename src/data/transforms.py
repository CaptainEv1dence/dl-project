from torchvision import transforms


def get_train_transforms(mean: float = 0.5, std: float = 0.5):
    """Augmenting transforms for the training split."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )


def get_eval_transforms(mean: float = 0.5, std: float = 0.5):
    """Deterministic transforms for validation / test splits."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )
