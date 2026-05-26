"""
One-time utility: convert folder-based FER-2013 images to the CSV format
expected by FER2013Dataset (columns: emotion, pixels, Usage).

Source layout (msambare/fer2013 from Kaggle):
    data/raw/train/{emotion}/*.jpg
    data/raw/test/{emotion}/*.jpg

Output:
    data/raw/fer2013.csv

The train/ folder is split 90% Training / 10% PublicTest (seed=42).
The test/ folder becomes PrivateTest.

Assumptions (verified against Kaggle folder names):
    angry    -> 0  (anger)
    disgust  -> 1
    fear     -> 2
    happy    -> 3  (happiness)
    sad      -> 4  (sadness)
    surprise -> 5
    neutral  -> 6
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

FOLDER_TO_LABEL: dict[str, int] = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}

IMAGE_SIZE = 48


def folder_to_rows(folder: Path, usage: str) -> list[dict]:
    rows: list[dict] = []
    for emotion_dir in sorted(folder.iterdir()):
        if not emotion_dir.is_dir():
            continue
        label = FOLDER_TO_LABEL.get(emotion_dir.name.lower())
        if label is None:
            print(f"  [WARN] unrecognised emotion folder: {emotion_dir.name} — skipped")
            continue
        for img_path in sorted(emotion_dir.iterdir()):
            try:
                img = Image.open(img_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
                pixels = np.array(img, dtype=np.uint8).flatten()
                pixel_str = " ".join(str(int(p)) for p in pixels)
                rows.append({"emotion": label, "pixels": pixel_str, "Usage": usage})
            except Exception as exc:
                print(f"  [WARN] could not read {img_path}: {exc} — skipped")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert folder FER-2013 to CSV.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing train/ and test/")
    parser.add_argument("--out", default="data/raw/fer2013.csv", help="Output CSV path")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of train to use as PublicTest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"

    for d in (train_dir, test_dir):
        if not d.exists():
            raise FileNotFoundError(f"Expected folder not found: {d}")

    print("Reading train/ ...")
    train_rows = folder_to_rows(train_dir, usage="Training")
    print(f"  {len(train_rows)} train images loaded")

    print("Reading test/ ...")
    test_rows = folder_to_rows(test_dir, usage="PrivateTest")
    print(f"  {len(test_rows)} test images loaded")

    # Split train 90/10 into Training / PublicTest
    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    val_n = int(len(train_rows) * args.val_frac)
    for row in train_rows[-val_n:]:
        row["Usage"] = "PublicTest"

    all_rows = train_rows + test_rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["emotion", "pixels", "Usage"])
        writer.writeheader()
        writer.writerows(all_rows)

    training_n = sum(1 for r in all_rows if r["Usage"] == "Training")
    public_n = sum(1 for r in all_rows if r["Usage"] == "PublicTest")
    private_n = sum(1 for r in all_rows if r["Usage"] == "PrivateTest")
    print(f"\nWrote {len(all_rows)} rows to {out_path}")
    print(f"  Training:    {training_n}")
    print(f"  PublicTest:  {public_n}")
    print(f"  PrivateTest: {private_n}")
    print("\n[NOTE] PublicTest split is derived from train/ (not original Kaggle PublicTest).")
    print("[NOTE] fer2013.csv is a derived artifact — do not commit to git.")


if __name__ == "__main__":
    main()
