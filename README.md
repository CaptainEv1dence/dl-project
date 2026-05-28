# Deep Emotion Recognition
<img width="1214" height="1122" alt="image" src="https://github.com/user-attachments/assets/90297ddf-62a1-47cc-8a86-afbb4058f57a" />

PyTorch pipeline for facial expression classification on FER-2013.

Implemented models:

* Baseline CNN
* SE-attention CNN
* ResNet-18 transfer learning
* EfficientNet-B2 transfer learning

The final main model is **EfficientNet-B2**.

---

## Results (test set, 7 178 samples)

| Model               |   Accuracy |   Macro-F1 | Weighted-F1 |
| ------------------- | ---------: | ---------: | ----------: |
| Baseline CNN        |      0.6108 |      0.5081 |       0.6031 |
| SE-CNN              |      0.6045 |      0.4966 |       0.5912 |
| ResNet-18           |      0.7034 |      0.6947 |       0.7028 |
| **EfficientNet-B2** | **0.7196** | **0.7147** |  **0.7192** |

EfficientNet-B2 is the best model.

EfficientNet-B2 uses ImageNet pretraining, 224x224 inputs, ImageNet normalization, stronger augmentation, label smoothing, sqrt class weights, AdamW, AMP, gradient clipping, and freeze/unfreeze fine-tuning.

Validation metrics for EfficientNet-B2:

| Split      | Accuracy | Macro-F1 | Weighted-F1 | Samples |
| ---------- | -------: | -------: | ----------: | ------: |
| Validation |   0.7129 |   0.6972 |      0.7135 |   2 870 |
| Test       |   0.7196 |   0.7147 |      0.7192 |   7 178 |

---

## Per-class F1 for EfficientNet-B2

| Emotion   | Precision | Recall |     F1 | Support |
| --------- | --------: | -----: | -----: | ------: |
| anger     |    0.6368 | 0.6534 | 0.6450 |     958 |
| disgust   |    0.8404 | 0.7117 | 0.7707 |     111 |
| fear      |    0.6231 | 0.5586 | 0.5891 |    1024 |
| happiness |    0.8999 | 0.8867 | 0.8932 |    1774 |
| sadness   |    0.6030 | 0.6079 | 0.6054 |    1247 |
| surprise  |    0.8128 | 0.8412 | 0.8267 |     831 |
| neutral   |    0.6510 | 0.6959 | 0.6727 |    1233 |

Strongest classes: **happiness** and **surprise**.

Hardest classes: **fear** and **sadness**.

---

## Per-class F1 for ResNet-18

| Emotion   | Precision | Recall |     F1 | Support |
| --------- | --------: | -----: | -----: | ------: |
| anger     |     0.636 |  0.621 |  0.629 |     958 |
| disgust   |     0.804 |  0.667 |  0.729 |     111 |
| fear      |     0.602 |  0.525 |  0.561 |    1024 |
| happiness |     0.887 |  0.890 |  0.888 |    1774 |
| sadness   |     0.563 |  0.589 |  0.575 |    1247 |
| surprise  |     0.824 |  0.824 |  0.824 |     831 |
| neutral   |     0.629 |  0.685 |  0.656 |    1233 |

Hardest class for ResNet-18: **fear** with F1 = 0.561, often confused with surprise and anger.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

For GPU training with CUDA 12.8 PyTorch wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If Streamlit or OpenCV is missing:

```bash
pip install streamlit opencv-python
```

---

## Dataset

The project expects FER-2013 in CSV format at:

```text
data/raw/fer2013.csv
```

Expected columns:

* `emotion` — class id from 0 to 6
* `pixels` — 2304 space-separated grayscale pixel values
* `Usage` — split name

### Option A — folder-based Kaggle download

Dataset: `msambare/fer2013`.

```bash
kaggle datasets download -d msambare/fer2013 -p data/raw --unzip
python -m src.data.convert_folders_to_csv --raw-dir data/raw --out data/raw/fer2013.csv
```

Note: the validation split is derived from the training folder using a 10% split with seed 42. The test folder is mapped to the test split.

### Validate and generate dataset stats

```bash
python -m src.data.prepare_data \
  --csv data/raw/fer2013.csv \
  --out data/processed/dataset_stats.json
```

---

## Training

### Baseline CNN

```bash
python -m src.train \
  --csv data/raw/fer2013.csv \
  --model baseline_cnn \
  --out outputs \
  --epochs 50 \
  --batch-size 128
```

### SE-CNN

```bash
python -m src.train \
  --csv data/raw/fer2013.csv \
  --model se_cnn \
  --out outputs \
  --epochs 50 \
  --batch-size 128
```

### ResNet-18

```bash
python -m src.train \
  --csv data/raw/fer2013.csv \
  --model resnet18 \
  --no-freeze-backbone \
  --out outputs \
  --epochs 30 \
  --batch-size 128 \
  --lr 3e-4
```

### EfficientNet-B2 final model

```bash
python -m src.train \
  --csv data/raw/fer2013.csv \
  --model efficientnet_b2 \
  --weights default \
  --freeze-backbone \
  --unfreeze-epoch 3 \
  --out outputs \
  --epochs 45 \
  --batch-size 32 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --image-size 224 \
  --imagenet-norm \
  --strong-aug \
  --label-smoothing 0.1 \
  --class-weights sqrt \
  --optimizer adamw \
  --amp \
  --grad-clip 1.0 \
  --num-workers 4
```

Training saves:

```text
outputs/checkpoints/best_<model>.pt
outputs/history_<model>.json
outputs/figures/curves_<model>.png
```

For EfficientNet-B2:

```text
outputs/checkpoints/best_efficientnet_b2.pt
outputs/history_efficientnet_b2.json
outputs/figures/curves_efficientnet_b2.png
```

---

## Evaluation

### ResNet-18 evaluation

```bash
python -m src.evaluate \
  --csv data/raw/fer2013.csv \
  --model resnet18 \
  --checkpoint outputs/checkpoints/best_resnet18.pt \
  --split test \
  --out outputs/metrics
```

### EfficientNet-B2 test evaluation

```bash
python -m src.evaluate \
  --csv data/raw/fer2013.csv \
  --model efficientnet_b2 \
  --checkpoint outputs/checkpoints/best_efficientnet_b2.pt \
  --split test \
  --out outputs/metrics \
  --image-size 224 \
  --imagenet-norm
```

### EfficientNet-B2 validation evaluation

```bash
python -m src.evaluate \
  --csv data/raw/fer2013.csv \
  --model efficientnet_b2 \
  --checkpoint outputs/checkpoints/best_efficientnet_b2.pt \
  --split val \
  --out outputs/metrics \
  --image-size 224 \
  --imagenet-norm
```

Evaluation saves:

```text
outputs/metrics/classification_report_<model>_<split>.json
outputs/metrics/confusion_matrix_<model>_<split>.png
```

For EfficientNet-B2:

```text
outputs/metrics/classification_report_efficientnet_b2_val.json
outputs/metrics/classification_report_efficientnet_b2_test.json
outputs/metrics/confusion_matrix_efficientnet_b2_val.png
outputs/metrics/confusion_matrix_efficientnet_b2_test.png
```

---

## Single-image prediction

### EfficientNet-B2

```bash
python -m src.predict \
  --checkpoint outputs/checkpoints/best_efficientnet_b2.pt \
  --model efficientnet_b2 \
  --image path/to/face.jpg \
  --image-size 224 \
  --imagenet-norm
```

### ResNet-18

```bash
python -m src.predict \
  --checkpoint outputs/checkpoints/best_resnet18.pt \
  --model resnet18 \
  --image path/to/face.jpg
```

---

## Mini dashboard and real-time inference

The project includes an interactive inference app:

* `app/inference.py` — shared model loading, preprocessing, and inference helpers
* `app/demo.py` — Streamlit mini-dashboard for image upload inference
* `app/realtime.py` — OpenCV webcam real-time emotion recognition

The app defaults to the final EfficientNet-B2 checkpoint:

```text
outputs/checkpoints/best_efficientnet_b2.pt
```

### Streamlit dashboard

Run:

```bash
streamlit run app/demo.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Dashboard features:

* upload a face image
* choose model, checkpoint, image size, normalization, and device
* show predicted emotion
* show confidence score
* show top predictions
* show probability bar chart
* show raw probability table

Default EfficientNet-B2 settings:

```text
checkpoint: outputs/checkpoints/best_efficientnet_b2.pt
model: efficientnet_b2
image_size: 224
imagenet_norm: true
device: auto
```

### Real-time webcam inference

Run:

```bash
python -m app.realtime \
  --checkpoint outputs/checkpoints/best_efficientnet_b2.pt \
  --model efficientnet_b2 \
  --image-size 224 \
  --imagenet-norm \
  --camera 0 \
  --device auto \
  --smoothing 5
```

Press `q` to close the webcam window.

If camera index `0` does not work, try camera index `1`:

```bash
python -m app.realtime \
  --checkpoint outputs/checkpoints/best_efficientnet_b2.pt \
  --model efficientnet_b2 \
  --image-size 224 \
  --imagenet-norm \
  --camera 1 \
  --device auto \
  --smoothing 5
```

Real-time pipeline:

1. capture webcam frame with OpenCV
2. detect face with Haar cascade
3. crop detected face
4. preprocess crop for EfficientNet-B2
5. run model inference
6. average probabilities over the last frames for smoother predictions
7. draw emotion label and confidence on the webcam frame

Notes:

* The dashboard works with uploaded images.
* The real-time app requires a working webcam.
* For best results, use a clearly visible frontal face.
* If CUDA is available, `--device auto` will use GPU automatically.
* If CUDA is not available, inference falls back to CPU.

---

## Project structure

```text
dl-project/
  src/
    config.py                    # labels, splits, dataclass configs
    train.py                     # training CLI
    evaluate.py                  # evaluation CLI
    predict.py                   # single-image inference
    data/
      prepare_data.py            # CSV validation and stats
      dataset.py                 # FER2013Dataset
      transforms.py              # train / eval transforms
      convert_folders_to_csv.py  # one-time folder to CSV converter
    models/
      baseline_cnn.py
      se_block.py
      se_cnn.py
      resnet18_transfer.py
      efficientnet_transfer.py
      __init__.py                # create_model() factory
    utils/
      metrics.py                 # accuracy, macro-F1, confusion matrix
      checkpoints.py             # save / load checkpoint
      plotting.py                # confusion matrix + training curves
      seed.py                    # deterministic seed
  app/
    inference.py                 # shared model loading and inference helpers
    demo.py                      # Streamlit dashboard for image upload inference
    realtime.py                  # OpenCV webcam real-time inference
  tests/                         # pytest suite
  data/raw/                      # .gitkeep only — dataset not committed
  data/processed/                # .gitkeep only — stats not committed
  outputs/
    checkpoints/                 # best model checkpoint
    figures/                     # training curves
    metrics/                     # classification reports and confusion matrices
  docs/templates/                # dataset readiness, experiment log, report outline
  requirements.txt
  TASK1_README.md
  TASK2_README.md
  TODO.md
```

---

## Running tests

```bash
pytest -q
```

Current status:

```text
19 passed
```

---

## Final EfficientNet-B2 artifacts

Generated after training and evaluation:

```text
outputs/checkpoints/best_efficientnet_b2.pt
outputs/history_efficientnet_b2.json
outputs/figures/curves_efficientnet_b2.png
outputs/metrics/classification_report_efficientnet_b2_val.json
outputs/metrics/classification_report_efficientnet_b2_test.json
outputs/metrics/confusion_matrix_efficientnet_b2_val.png
outputs/metrics/confusion_matrix_efficientnet_b2_test.png
```

---

