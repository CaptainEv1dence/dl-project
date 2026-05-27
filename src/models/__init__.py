from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet_transfer import EfficientNetB2EmotionClassifier
from src.models.resnet18_transfer import ResNet18EmotionClassifier
from src.models.se_cnn import SECNN

VALID_MODELS = ("baseline_cnn", "se_cnn", "resnet18", "efficientnet_b2")


def create_model(model_name: str, num_classes: int = 7, **model_kwargs):
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)

    if model_name == "se_cnn":
        return SECNN(num_classes=num_classes)

    if model_name == "resnet18":
        return ResNet18EmotionClassifier(num_classes=num_classes, **model_kwargs)

    if model_name == "efficientnet_b2":
        return EfficientNetB2EmotionClassifier(num_classes=num_classes, **model_kwargs)

    raise ValueError(f"Unknown model_name: {model_name!r}. Valid: {', '.join(VALID_MODELS)}")
