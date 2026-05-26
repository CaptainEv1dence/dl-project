import torch
from src.data.dataset import FER2013Dataset
from src.data.transforms import get_eval_transforms


def test_dataset_filters_split_and_returns_tensor(sample_fer_csv):
    dataset = FER2013Dataset(sample_fer_csv, split="train", transform=get_eval_transforms(mean=0.0, std=1.0))
    image, label = dataset[0]
    assert len(dataset) == 1
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 48, 48)
    assert label == 0
    assert 0.0 <= float(image.min()) <= 1.0
    assert 0.0 <= float(image.max()) <= 1.0


def test_dataset_raises_for_unknown_split(sample_fer_csv):
    try:
        FER2013Dataset(sample_fer_csv, split="dev")
    except ValueError as exc:
        assert "split" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for unknown split")
