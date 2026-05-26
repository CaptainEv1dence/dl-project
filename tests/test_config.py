from src.config import EMOTION_LABELS, NUM_CLASSES, SPLIT_TO_USAGE


def test_emotion_labels_match_fer2013_order():
    assert NUM_CLASSES == 7
    assert EMOTION_LABELS == {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "happiness",
        4: "sadness",
        5: "surprise",
        6: "neutral",
    }


def test_split_to_usage_mapping():
    assert SPLIT_TO_USAGE == {
        "train": "Training",
        "val": "PublicTest",
        "test": "PrivateTest",
    }
