import pytest
import torch
from src.models import create_model
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet18_transfer import ResNet18EmotionClassifier
from src.models.se_block import SEBlock
from src.models.se_cnn import SECNN


def test_baseline_cnn_forward_shape():
    model = BaselineCNN(num_classes=7)
    x = torch.randn(2, 1, 48, 48)
    logits = model(x)
    assert logits.shape == (2, 7)


def test_se_block_preserves_feature_shape():
    block = SEBlock(channels=32, reduction=8)
    x = torch.randn(2, 32, 24, 24)
    y = block(x)
    assert y.shape == x.shape


def test_se_cnn_forward_shape():
    model = SECNN(num_classes=7)
    x = torch.randn(2, 1, 48, 48)
    logits = model(x)
    assert logits.shape == (2, 7)


def test_resnet18_accepts_grayscale_and_outputs_logits():
    model = ResNet18EmotionClassifier(num_classes=7, weights=None, freeze_backbone=True)
    x = torch.randn(2, 1, 48, 48)
    logits = model(x)
    assert logits.shape == (2, 7)


def test_resnet18_freezes_backbone_but_not_classifier():
    model = ResNet18EmotionClassifier(num_classes=7, weights=None, freeze_backbone=True)
    backbone_params = [p.requires_grad for name, p in model.backbone.named_parameters() if not name.startswith("fc.")]
    head_params = [p.requires_grad for name, p in model.backbone.named_parameters() if name.startswith("fc.")]
    assert backbone_params
    assert head_params
    assert not any(backbone_params)
    assert all(head_params)


def test_create_model_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown model_name"):
        create_model("bad_name")
