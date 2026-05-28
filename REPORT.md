# Deep Emotion Recognition — Project Report

**Course project:** Facial expression classification on FER-2013  
**Team:** Iaroslav Kolomiets, Elizaveta Kamenskaya, Petr Kovalev, Dmitrii Plotnikov

---

## 1. Motivation

Recognizing facial expressions from images is a core building block of **affective computing**: systems that infer emotional state from visual cues. Applications include human–robot interaction, accessibility tools, driver monitoring, and lightweight UX analytics. The FER-2013 benchmark provides a standardized, challenging setting: **seven basic emotions** from **low-resolution 48×48 grayscale** crops, with strong **class imbalance** (e.g. disgust is ~16× rarer than happiness).

Our goal was to build a reproducible PyTorch pipeline—from data validation through training, evaluation, and deployment—and to **progressively improve macro-F1** beyond simple CNN baselines, ending with a model suitable for **interactive demo and real-time webcam inference**.

---

## 2. Dataset and Preprocessing

### 2.1 Source and format


| Item             | Detail                                                                            |
| ---------------- | --------------------------------------------------------------------------------- |
| Dataset          | [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) (`msambare/fer2013`) |
| Local path       | `data/raw/fer2013.csv`                                                            |
| Schema           | `emotion` (0–6), `pixels` (2304 values), `Usage`                                  |
| Total valid rows | 35,887                                                                            |


Folder-based Kaggle data were converted once with:

```bash
python -m src.data.convert_folders_to_csv --raw-dir data/raw --out data/raw/fer2013.csv
```

**Split policy:** `Training` uses the train folder; `PrivateTest` uses the test folder. **Validation** (`PublicTest`) is a **10% hold-out from train** (seed 42, `random_state=42`), not the original Kaggle competition public split. This keeps a fixed validation set for checkpoint selection while using the Kaggle `test/` folder as the held-out test set.

### 2.2 Split and class distribution


| Split      | Usage column  | Samples |
| ---------- | ------------- | ------- |
| Train      | `Training`    | 25,839  |
| Validation | `PublicTest`  | 2,870   |
| Test       | `PrivateTest` | 7,178   |



| Emotion   | Train+val+test count | Share (approx.) |
| --------- | -------------------- | --------------- |
| anger     | 4,953                | 13.8%           |
| disgust   | 547                  | 1.5%            |
| fear      | 5,121                | 14.3%           |
| happiness | 8,989                | 25.0%           |
| sadness   | 6,077                | 16.9%           |
| surprise  | 4,002                | 11.2%           |
| neutral   | 6,198                | 17.3%           |


(docs/figures/class distrib.png)
(docs/figures/class distrib train val.png)
(docs/figures/sample.png)

Pixel statistics (grayscale): mean **129.38**, std **65.08**.

**Why macro-F1:** Accuracy is dominated by frequent classes (happiness). We select checkpoints and compare models by **validation macro-F1**, then report test metrics once for the chosen model.

### 2.3 Preprocessing pipeline

Implemented in `src/data/dataset.py` and `src/data/transforms.py`:

- Parse `pixels` -> 48×48 tensor, scale to [0, 1].
- **Baseline models:** 48×48, single channel, normalize with mean=0.5, std=0.5.
- **EfficientNet-B2:** resize to **224×224**, replicate to 3 channels, **ImageNet** mean/std.
- **Train-only augmentation:** horizontal flip, rotation (±12°), affine jitter, optional **ColorJitter** (brightness/contrast) and **RandomErasing** (`--strong-aug`).
- Validation/test: deterministic resize + normalize only.

Data readiness is validated with:

```bash
python -m src.data.prepare_data --csv data/raw/fer2013.csv --out data/processed/dataset_stats.json
```

--

## 4. Methods

We implemented four model families behind a single factory (`src/models/__init__.py` -> `create_model()`).

### 4.1 Baseline CNN

Three conv blocks (32->64->128 channels), batch norm, max pooling, MLP head with dropout. Trained from scratch at **48×48** without ImageNet weights. Serves as a simple, fully custom baseline.

### 4.2 SE-CNN

Same backbone as the baseline with **Squeeze-and-Excitation** blocks (`src/models/se_block.py`, `src/models/se_cnn.py`) to re-weight channel responses. Hypothesis: attention helps on noisy, low-res faces.

### 4.3 ResNet-18 transfer learning

TorchVision `ResNet18_Weights.DEFAULT`, grayscale -> 3-channel repeat, replaced final linear for 7 classes (`src/models/resnet18_transfer.py`). Establishes a strong transfer-learning baseline at 48×48 (or configurable size).

### 4.4 EfficientNet-B2 (final model)

ImageNet-pretrained EfficientNet-B2 (`src/models/efficientnet_transfer.py`):

- Grayscale input repeated to 3 channels.
- Custom classifier head: Dropout(0.3) + Linear(7).
- **Two-phase fine-tuning:** backbone frozen for epochs 0–2, full unfreeze from epoch 3 (`--freeze-backbone --unfreeze-epoch 3`).
- Input **224×224**, ImageNet normalization.

**Training recipe (final run):**


| Hyperparameter   | Value                                  |
| ---------------- | -------------------------------------- |
| Optimizer        | AdamW                                  |
| Learning rate    | 3×10^-4                                |
| Weight decay     | 10^-4                                  |
| Scheduler        | CosineAnnealingLR (45 epochs)          |
| Batch size       | 32                                     |
| Loss             | CrossEntropy + label smoothing 0.1     |
| Class weights    | sqrt of balanced weights               |
| AMP              | enabled                                |
| Gradient clip    | 1.0                                    |
| Selection metric | Validation   macro-F1                  |


**Rationale:** Literature and our ResNet-18 results showed transfer learning beats from-scratch CNNs on FER-2013. EfficientNet-B2 offers a better accuracy–efficiency trade-off than ResNet-18. Label smoothing and sqrt class weights mitigate overconfidence and imbalance without over-penalizing majority classes. Freeze-then-unfreeze stabilizes the new head before updating low-level filters.

---

## 5. Work Performed (Pipeline and Engineering)


| Step       | What we did                                                                                               | Why                                            |
| ---------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Data layer | CSV loader, split filtering, transforms, folder->CSV converter                                            | Reproducible FER-2013 ingestion                |
| Training   | Unified `src/train.py` CLI for all models; checkpoint by val macro-F1; history JSON + curve plots         | Fair comparison, traceability                  |
| Metrics    | Accuracy, macro/weighted F1, per-class report, confusion matrices (`src/utils/metrics.py`, `plotting.py`) | Imbalance-aware evaluation                     |
| Evaluation | `src/evaluate.py` for val/test splits                                                                     | Standardized reports after training            |
| Inference  | `src/predict.py` single-image CLI                                                                         | Quick sanity checks                            |
| Apps       | Streamlit `app/demo.py`, OpenCV `app/realtime.py`, shared `app/inference.py`                              | Demonstrate deployment beyond offline test set |
| Tests      | 19 pytest tests (dataset, models, metrics, smoke train/eval)                                              | Catch regressions in CI/local runs             |


**Training commands (representative):**

```bash
# Baseline / SE-CNN
python -m src.train --csv data/raw/fer2013.csv --model baseline_cnn --out outputs --epochs 50 --batch-size 128
python -m src.train --csv data/raw/fer2013.csv --model se_cnn --out outputs --epochs 50 --batch-size 128

# ResNet-18
python -m src.train --csv data/raw/fer2013.csv --model resnet18 --no-freeze-backbone \
  --out outputs --epochs 30 --batch-size 128 --lr 3e-4

# EfficientNet-B2 (final)
python -m src.train --csv data/raw/fer2013.csv --model efficientnet_b2 --weights default \
  --freeze-backbone --unfreeze-epoch 3 --out outputs --epochs 45 --batch-size 32 \
  --lr 3e-4 --weight-decay 1e-4 --image-size 224 --imagenet-norm --strong-aug \
  --label-smoothing 0.1 --class-weights sqrt --optimizer adamw --amp --grad-clip 1.0 --num-workers 4
```

---

## 6. Results

### 6.1 Model comparison (test set, n = 7,178)

| Model               | Accuracy   | Macro-F1   | Weighted-F1 |
| ------------------- | ---------- | ---------- | ----------- |
| Baseline CNN        | 0.592      | 0.488      | 0.581       |
| SE-CNN              | 0.603      | 0.533      | 0.599       |
| ResNet-18           | 0.637      | 0.633      | 0.636       |
| **EfficientNet-B2** | **0.7196** | **0.7147** | **0.7192**  |


### 6.2 EfficientNet-B2 — validation vs test


| Split      | Samples | Accuracy | Macro-F1 | Weighted-F1 |
| ---------- | ------- | -------- | -------- | ----------- |
| Validation | 2,870   | 0.7129   | 0.6972   | 0.7135      |
| Test       | 7,178   | 0.7196   | 0.7147   | 0.7192      |

Best checkpoint saved as `outputs/checkpoints/best_efficientnet_b2.pt`.

### 6.3 Per-class performance (EfficientNet-B2, test)


| Emotion   | Precision | Recall | F1        | Support |
| --------- | --------- | ------ | --------- | ------- |
| anger     | 0.637     | 0.653  | 0.645     | 958     |
| disgust   | 0.840     | 0.712  | 0.771     | 111     |
| fear      | 0.623     | 0.559  | 0.589     | 1,024   |
| happiness | 0.900     | 0.887  | **0.893** | 1,774   |
| sadness   | 0.603     | 0.608  | 0.605     | 1,247   |
| surprise  | 0.813     | 0.841  | **0.827** | 831     |
| neutral   | 0.651     | 0.696  | 0.673     | 1,233   |


### 6.4 ResNet-18 per-class F1 (test, reference)


| Emotion   | F1    | Notes                            |
| --------- | ----- | -------------------------------- |
| fear      | 0.476 | Main bottleneck vs EfficientNet  |
| sadness   | 0.500 | Often confused with fear/neutral |
| happiness | 0.827 | Already strong                   |
| surprise  | 0.775 | —                                |

EfficientNet-B2’s largest gain on fear (+0.113 F1) and sadness (+0.105 F1) explains most of the macro-F1 improvement.

### 6.5 Training dynamics (EfficientNet-B2)

(outputs/figures/curves_efficientnet_b2.png)

After unfreezing (epoch 3), validation macro-F1 rises sharply—pretraining provides useful features once the head has warmed up. Later epochs show mild overfitting on train accuracy (~0.99) while validation plateaus near ~0.71 accuracy / ~0.70 macro-F1.

### 6.6 Confusion matrices

**Test set** (rows = true label, columns = predicted; order: anger, disgust, fear, happiness, sadness, surprise, neutral):

Confusion matrix — EfficientNet-B2, test split
(docs/figures/confusion_matrix_efficientnet_b2_test.png)

**Validation split:**

Confusion matrix — EfficientNet-B2, validation split
(docs/figures/confusion_matrix_efficientnet_b2_val.png)

---

## 7. Deployment: Demo and Real-Time Recognition

### 7.1 Streamlit dashboard

`streamlit run app/demo.py` — upload an image, select model/checkpoint, view top-k probabilities and bar chart. Defaults point to EfficientNet-B2 at 224px with ImageNet normalization.

### 7.2 Real-time webcam pipeline

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

Pipeline: OpenCV capture -> **Haar frontal-face detector** -> crop -> same preprocessing as offline inference -> EfficientNet-B2 -> **moving average over 5 frames** for stable labels. Runs on **CUDA** when available (see terminal logs in screenshots).

### 7.3 Real-time detection examples

The model was tested live on a laptop webcam. Below: bounding box, emotion label, and confidence.

**Happiness (87.7% confidence):**
(docs/figures/realtime/happiness_87.7.png)

**Fear (84.2% confidence):**
(docs/figures/realtime/fear_84.2.png)

---

## 8. Findings and Conclusions

1. **Transfer learning at higher resolution wins on FER-2013.** EfficientNet-B2 at 224px with ImageNet pretraining and a disciplined fine-tuning recipe reaches **~72% test accuracy** and **~0.715 macro-F1**.
2. **Macro-F1 and class weights are essential** given imbalance: sqrt-weighted loss + label smoothing improved minority-class behavior without collapsing majority-class precision.
3. **Freeze-then-unfreeze is effective:** most gain appears at epoch 3 when the backbone unfreezes, best val macro-F1 occurs mid-schedule (epoch 32), not the last epoch—early stopping on val macro-F1 is justified.
---

## 10. Reproducibility

**Evaluate final model:**

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