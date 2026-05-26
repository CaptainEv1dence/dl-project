# Deep Emotion Recognition

PyTorch pipeline for facial expression classification on FER-2013.
Three models: baseline CNN, SE-attention CNN, ResNet-18 transfer learning.

---

## Results (test set, 7 178 samples)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Baseline CNN | 0.592 | 0.488 | 0.581 |
| SE-CNN | 0.603 | 0.533 | 0.599 |
| **ResNet-18** | **0.637** | **0.633** | **0.636** |

ResNet-18 (all layers fine-tuned, lr=3e-4, 30 epochs, cosine LR) is the best model.

Per-class F1 for ResNet-18:

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.526 | 0.554 | 0.540 | 958 |
| disgust | 0.802 | 0.622 | 0.701 | 111 |
| fear | 0.541 | 0.425 | 0.476 | 1024 |
| happiness | 0.841 | 0.813 | 0.827 | 1774 |
| sadness | 0.476 | 0.526 | 0.500 | 1247 |
| surprise | 0.746 | 0.807 | 0.775 | 831 |
| neutral | 0.598 | 0.623 | 0.610 | 1233 |

Hardest class: **fear** (F1=0.476), often confused with surprise and anger.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. For GPU (CUDA 12.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

---

## Dataset

The project expects FER-2013 in CSV format at `data/raw/fer2013.csv`
with columns: `emotion` (0–6), `pixels` (2304 space-separated values), `Usage`.

**Option A — folder-based Kaggle download** (`msambare/fer2013`):

```bash
kaggle datasets download -d msambare/fer2013 -p data/raw --unzip
python -m src.data.convert_folders_to_csv --raw-dir data/raw --out data/raw/fer2013.csv
```

> Note: the val split (PublicTest) is derived from train/ (10%, seed=42),
> not from the original Kaggle competition split.

**Validate and generate stats:**

```bash
python -m src.data.prepare_data --csv data/raw/fer2013.csv --out data/processed/dataset_stats.json
```

---

## Training

```bash
# Baseline CNN
python -m src.train --csv data/raw/fer2013.csv --model baseline_cnn --out outputs --epochs 50 --batch-size 128

# SE-CNN
python -m src.train --csv data/raw/fer2013.csv --model se_cnn --out outputs --epochs 50 --batch-size 128

# ResNet-18 (full fine-tune, lower LR)
python -m src.train --csv data/raw/fer2013.csv --model resnet18 --no-freeze-backbone --out outputs --epochs 30 --batch-size 128 --lr 3e-4
```

Saves: `outputs/checkpoints/best_<model>.pt`, `outputs/history_<model>.json`, `outputs/figures/curves_<model>.png`

---

## Evaluation

```bash
python -m src.evaluate \
    --csv data/raw/fer2013.csv \
    --model resnet18 \
    --checkpoint outputs/checkpoints/best_resnet18.pt \
    --split test \
    --out outputs/metrics
```

Saves: `classification_report_<split>.json`, `confusion_matrix_<split>.png`

---

## Single-image prediction

```bash
python -m src.predict \
    --checkpoint outputs/checkpoints/best_resnet18.pt \
    --model resnet18 \
    --image path/to/face.jpg
```

---

## Project structure

```
dl-project/
  src/
    config.py               # labels, splits, dataclass configs
    train.py                # training CLI
    evaluate.py             # evaluation CLI
    predict.py              # single-image inference
    data/
      prepare_data.py       # CSV validation and stats
      dataset.py            # FER2013Dataset
      transforms.py         # train / eval transforms
      convert_folders_to_csv.py  # one-time folder→CSV converter
    models/
      baseline_cnn.py
      se_block.py
      se_cnn.py
      resnet18_transfer.py
      __init__.py           # create_model() factory
    utils/
      metrics.py            # accuracy, macro-F1, confusion matrix
      checkpoints.py        # save / load checkpoint
      plotting.py           # confusion matrix + training curves
      seed.py               # deterministic seed
  tests/                    # pytest suite (16 tests)
  app/                      # demo stubs (see TODO)
  data/raw/                 # .gitkeep only — dataset not committed
  data/processed/           # .gitkeep only — stats not committed
  outputs/
    checkpoints/            # .gitkeep only — .pt files not committed
    figures/                # .gitkeep only
    metrics/                # .gitkeep only
  docs/templates/           # dataset readiness, experiment log, report outline
  requirements.txt
```

---

## Running tests

```bash
pytest tests/ -v
```

16 tests, all passing.

---

## What's next

See [TODO.md](TODO.md) for the full list. Key open items:

- **Better model**: implement EfficientNet-B2 — literature reports ~70-73% accuracy on FER-2013 vs our 63.7% with ResNet-18
- **Streamlit demo**: `app/demo.py` — image upload, emotion label, probability bar chart
- **Real-time webcam**: `app/realtime.py` — OpenCV loop with Haar cascade face detection
- **Training**: label smoothing, weighted loss for disgust class imbalance, stronger augmentation
