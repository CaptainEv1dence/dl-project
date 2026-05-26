# Report outline

## 1. Introduction

- Problem: seven-class facial expression classification from 48x48 grayscale FER-2013 images.
- Motivation: affective computing, human-robot interaction, user experience analysis.
- Scope: baseline CNN, SE-CNN, ResNet-18 transfer learning, metrics, optional demo.

## 2. Dataset and preprocessing

- Dataset identity and source from `docs/templates/dataset_readiness.md` copy.
- Schema: `emotion`, `pixels`, `Usage`.
- Cleaning rules: 2304 pixels, labels 0-6, valid splits, pixel values 0-255.
- Derived artifacts: `data/processed/dataset_stats.json`, class distribution plot.
- Note class imbalance, especially disgust.

## 3. Methods

### 3.1 Baseline CNN

- Three convolution blocks, batch norm, ReLU, max pooling, dropout classifier.

### 3.2 SE-CNN

- Same CNN backbone plus Squeeze-and-Excitation blocks.

### 3.3 ResNet-18 transfer learning

- Repeat grayscale channel to RGB.
- Use TorchVision `ResNet18_Weights.DEFAULT`.
- Replace classifier with seven-class head.

## 4. Experimental setup

- Splits: `Training`, `PublicTest`, `PrivateTest`.
- Main selection metric: validation macro-F1.
- Final test evaluated once after model selection.
- Augmentations used only on training split.

## 5. Results

- Fill from `docs/templates/results_table.csv` copy.
- Include confusion matrix for best model.
- Include train/validation curves.

## 6. Error analysis

- Most common confusion pairs.
- Example misclassifications.
- Likely causes: class ambiguity, low resolution, occlusion, lighting, class imbalance.

## 7. Demo and optional real-time recognition

- Upload demo pipeline.
- Webcam pipeline if implemented.
- Mention prediction smoothing if used.

## 8. Limitations and future work

- FER-2013 low resolution.
- Domain gap between benchmark images and webcam frames.
- Seven basic classes only.
- Future work: better face detector, larger dataset, temporal video model, robot-dog deployment.
