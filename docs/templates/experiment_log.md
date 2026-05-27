# Experiment log

Create one copy per training run. Keep commands and outputs exact.

## Run identity

- Run name: `[model-date-seed]`
- Owner: `[name]`
- Date: `[YYYY-MM-DD]`
- Model: `[baseline_cnn | se_cnn | resnet18]`
- Dataset checksum: `[sha256 from dataset_readiness.md]`
- Seed: `[seed]`

## Training command

```bash
[exact training command]
```

## Environment

- Python version: `[version]`
- PyTorch version: `[version]`
- TorchVision version: `[version]`
- Device: `[cpu | cuda device name]`

## Key outputs

- Checkpoint path: `[path]`
- Metrics file: `[path]`
- Figure paths: `[paths]`

## Validation metrics

| Metric | Value |
|---|---:|
| accuracy | `[value]` |
| macro-F1 | `[value]` |
| weighted-F1 | `[value]` |

## Test metrics

| Metric | Value |
|---|---:|
| accuracy | `[value]` |
| macro-F1 | `[value]` |
| weighted-F1 | `[value]` |

## Notes

- Observed failure modes: `[notes]`
- Confusing class pairs: `[notes]`
- Next run change: `[one change only]`

## EfficientNet-B2 transfer learning

### Setup

- Backbone: EfficientNet-B2
- Pretraining: ImageNet
- Input size: 224x224
- Normalization: ImageNet mean/std
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 1e-4
- Scheduler: CosineAnnealingLR
- Epochs: 45
- Batch size: 32
- Loss: CrossEntropyLoss with label_smoothing=0.1 and sqrt class weights
- Augmentations: horizontal flip, rotation, affine transform, brightness/contrast jitter, random erasing
- Fine-tuning: frozen backbone for first 3 epochs, then full unfreeze
- AMP: enabled
- Gradient clipping: 1.0

### Validation results

- Accuracy: 0.7129
- Macro-F1: 0.6972
- Weighted-F1: 0.7135
- Samples: 2870

### Test results

- Accuracy: 0.7196
- Macro-F1: 0.7147
- Weighted-F1: 0.7192
- Samples: 7178

### Per-class test F1

| Class | F1-score |
|---|---:|
| anger | 0.6450 |
| disgust | 0.7707 |
| fear | 0.5891 |
| happiness | 0.8932 |
| sadness | 0.6054 |
| surprise | 0.8267 |
| neutral | 0.6727 |

### Notes

EfficientNet-B2 improves over the previous baseline models and becomes the main model for the final submission. The strongest classes are happiness and surprise. The weakest classes are fear and sadness, which are often confused with visually similar negative or neutral emotions.

### Artifacts

- outputs/checkpoints/best_efficientnet_b2.pt
- outputs/history_efficientnet_b2.json
- outputs/figures/curves_efficientnet_b2.png
- outputs/metrics/classification_report_efficientnet_b2_val.json
- outputs/metrics/classification_report_efficientnet_b2_test.json
- outputs/metrics/confusion_matrix_efficientnet_b2_val.png
- outputs/metrics/confusion_matrix_efficientnet_b2_test.png
