from src.models.baseline_cnn import BaselineCNN
from src.models.resnet18_transfer import ResNet18EmotionClassifier
from src.models.se_cnn import SECNN


def create_model(model_name: str, num_classes: int = 7, **model_kwargs):
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "se_cnn":
        return SECNN(num_classes=num_classes)
    if model_name == "resnet18":
        return ResNet18EmotionClassifier(num_classes=num_classes, **model_kwargs)
    raise ValueError(f"Unknown model_name: {model_name!r}. Valid: baseline_cnn, se_cnn, resnet18")
