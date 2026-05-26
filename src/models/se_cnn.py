import torch
import torch.nn as nn

from src.models.se_block import SEBlock


class SECNN(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.5, reduction: int = 16) -> None:
        super().__init__()
        # Block 1: 1x48x48 -> 32x24x24 -> SE(32)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.se1 = SEBlock(32, reduction=reduction)

        # Block 2: 32x24x24 -> 64x12x12 -> SE(64)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.se2 = SEBlock(64, reduction=reduction)

        # Block 3: 64x12x12 -> 128x6x6 -> SE(128)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.se3 = SEBlock(128, reduction=reduction)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se1(self.block1(x))
        x = self.se2(self.block2(x))
        x = self.se3(self.block3(x))
        return self.classifier(x)
