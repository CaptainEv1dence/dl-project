# Task 1: EfficientNet-B2 + stronger training

## Files in this patch

- `src/models/efficientnet_transfer.py`
- `src/models/__init__.py`
- `src/data/transforms.py`
- `src/train.py`
- `src/evaluate.py`
- `src/predict.py`
- `tests/test_models.py.patch_snippet`

`tests/test_models.py.patch_snippet` is not an executable patch; copy its import/tests into the existing `tests/test_models.py`.

## Smoke test

```bash
pytest tests/test_models.py -q
python -m src.train \
  --csv data/raw/fer2013.csv \
  --model efficientnet_b2 \
  --weights none \
  --out outputs/smoke \
  --epochs 1 \
  --batch-size 8 \
  --image-size 64 \
  --strong-aug \
  --label-smoothing 0.1 \
  --class-weights sqrt
```

## Main training run

Use pretrained ImageNet weights if the machine has internet or cached torchvision weights:

```bash
python -m src.train \
  --csv data/raw/fer2013.csv \
  --model efficientnet_b2 \
  --weights default \
  --freeze-backbone \
  --unfreeze-epoch 3 \
  --out outputs \
  --epochs 45 \
  --batch-size 64 \
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

If VRAM is low, use `--batch-size 32`. If it is still slow, use `--image-size 160`.

## Evaluation

```bash
python -m src.evaluate \
  --csv data/raw/fer2013.csv \
  --model efficientnet_b2 \
  --checkpoint outputs/checkpoints/best_efficientnet_b2.pt \
  --split val \
  --out outputs/metrics \
  --image-size 224 \
  --imagenet-norm

python -m src.evaluate \
  --csv data/raw/fer2013.csv \
  --model efficientnet_b2 \
  --checkpoint outputs/checkpoints/best_efficientnet_b2.pt \
  --split test \
  --out outputs/metrics \
  --image-size 224 \
  --imagenet-norm
```

## Metrics to report

- val best macro-F1 from `outputs/history_efficientnet_b2.json`
- test accuracy, macro-F1, weighted-F1 from `outputs/metrics/classification_report_efficientnet_b2_test.json`
- per-class precision/recall/F1/support from the same JSON
- confusion matrix PNG: `outputs/metrics/confusion_matrix_efficientnet_b2_test.png`
- training curves PNG: `outputs/figures/curves_efficientnet_b2.png`
