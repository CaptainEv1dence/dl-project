import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EMOTION_LABELS

REQUIRED_COLUMNS = {"emotion", "pixels", "Usage"}
VALID_USAGES = {"Training", "PublicTest", "PrivateTest"}
PIXEL_COUNT = 48 * 48


def _parse_pixels(raw_pixels: object) -> np.ndarray | None:
    if not isinstance(raw_pixels, str):
        return None

    parts = raw_pixels.split()
    if len(parts) != PIXEL_COUNT:
        return None

    try:
        pixels = np.array([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None

    if np.any(pixels < 0) or np.any(pixels > 255):
        return None

    return pixels


def _parse_label(raw_label: object) -> int | None:
    try:
        label = int(raw_label)
    except (TypeError, ValueError):
        return None

    if label not in EMOTION_LABELS:
        return None
    return label


def validate_fer2013_csv(csv_path: str | Path) -> dict:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"FER-2013 CSV not found: {path}")

    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"FER-2013 CSV missing required columns: {missing}")

    split_counts: dict[str, int] = {}
    class_counts = {emotion: 0 for emotion in EMOTION_LABELS.values()}
    valid_pixels: list[np.ndarray] = []
    removed_rows = 0

    for row in df.itertuples(index=False):
        row_data = row._asdict()
        label = _parse_label(row_data["emotion"])
        usage = row_data["Usage"]
        pixels = _parse_pixels(row_data["pixels"])

        if label is None or usage not in VALID_USAGES or pixels is None:
            removed_rows += 1
            continue

        split_counts[usage] = split_counts.get(usage, 0) + 1
        class_counts[EMOTION_LABELS[label]] += 1
        valid_pixels.append(pixels)

    if valid_pixels:
        all_pixels = np.concatenate(valid_pixels)
        pixel_mean = float(all_pixels.mean())
        pixel_std = float(all_pixels.std())
    else:
        pixel_mean = 0.0
        pixel_std = 0.0

    return {
        "source_csv": str(path),
        "valid_rows": len(valid_pixels),
        "removed_rows": removed_rows,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "derived_artifact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FER-2013 CSV and write derived dataset stats.")
    parser.add_argument("--csv", required=True, help="Path to FER-2013 CSV file.")
    parser.add_argument("--out", required=True, help="Path for derived dataset statistics JSON.")
    args = parser.parse_args()

    stats = validate_fer2013_csv(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote derived dataset stats to {out_path}")


if __name__ == "__main__":
    main()
