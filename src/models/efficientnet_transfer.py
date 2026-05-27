import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


class EfficientNetB2EmotionClassifier(nn.Module):
    """EfficientNet-B2 transfer-learning classifier for FER-2013 emotions.

    Accepts either (B, 1, H, W) grayscale tensors or (B, 3, H, W) RGB tensors.
    If input is grayscale, it is repeated to 3 channels before the ImageNet backbone.
    """

    def __init__(
        self,
        num_classes: int = 7,
        weights="default",
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if weights == "default":
            weights_arg = EfficientNet_B2_Weights.DEFAULT
        else:
            # None or an explicit torchvision Weights enum/string.
            weights_arg = weights

        self.backbone = efficientnet_b2(weights=weights_arg)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
