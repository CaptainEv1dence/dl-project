from dataclasses import dataclass

NUM_CLASSES = 7

EMOTION_LABELS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

SPLIT_TO_USAGE = {
    "train": "Training",
    "val": "PublicTest",
    "test": "PrivateTest",
}


@dataclass(frozen=True)
class DataConfig:
    csv_path: str = "data/raw/fer2013.csv"
    image_size: int = 48


@dataclass(frozen=True)
class TrainConfig:
    model: str = "baseline_cnn"
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
