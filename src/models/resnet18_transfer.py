import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18EmotionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        weights="default",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if weights == "default":
            weights_arg = ResNet18_Weights.DEFAULT
        else:
            # None or an explicit Weights enum value
            weights_arg = weights

        self.backbone = resnet18(weights=weights_arg)
        # Replace the final fully-connected layer
        self.backbone.fc = nn.Linear(512, num_classes)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
