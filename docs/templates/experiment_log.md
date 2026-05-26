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
