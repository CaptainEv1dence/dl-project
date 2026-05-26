from pathlib import Path

import pandas as pd
import pytest


def pixel_string(value: int = 0, length: int = 2304) -> str:
    return " ".join([str(value)] * length)


@pytest.fixture
def sample_fer_csv(tmp_path: Path) -> Path:
    path = tmp_path / "fer2013.csv"
    df = pd.DataFrame(
        [
            {"emotion": 0, "pixels": pixel_string(10), "Usage": "Training"},
            {"emotion": 3, "pixels": pixel_string(20), "Usage": "PublicTest"},
            {"emotion": 6, "pixels": pixel_string(30), "Usage": "PrivateTest"},
            {"emotion": 1, "pixels": pixel_string(40, length=10), "Usage": "Training"},
        ]
    )
    df.to_csv(path, index=False)
    return path
