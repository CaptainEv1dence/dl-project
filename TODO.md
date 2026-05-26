# TODO

## New model — beat ResNet-18 (macro-F1 0.633)

Research shows EfficientNet-B2 and Swin-T outperform ResNet-18 on FER-2013
(EfficientNet-B2: ~70-73% accuracy in literature, Swin-FER: 71.11%).

- [ ] Add `src/models/efficientnet_transfer.py` — `EfficientNetB2Emotion` class
      (same pattern as `resnet18_transfer.py`: grayscale→3ch, replace classifier head)
- [ ] Register in `create_model()` factory as `"efficientnet_b2"`
- [ ] Add `tests/test_models.py` shape test for EfficientNet-B2
- [ ] Train with label smoothing (`CrossEntropyLoss(label_smoothing=0.1)`)
      and stronger augmentation (RandomErasing, ColorJitter) — add to `train.py`
- [ ] Consider Swin-T (`torchvision.models.swin_t`) as alternative — 28M params
- [ ] Compare val macro-F1 against ResNet-18 baseline (0.604)
- [ ] Run full evaluation on test split and report numbers

## Demo — image upload UI

- [ ] Implement `app/demo.py` using Streamlit
  - model selector (dropdown: baseline_cnn / se_cnn / resnet18 / efficientnet_b2)
  - checkpoint path input
  - image uploader
  - show uploaded image + predicted emotion label
  - probability bar chart per class
  - reuse `predict_image()` from `src/predict.py` — no duplicate preprocessing

## Real-time webcam recognition

- [ ] Implement `app/realtime.py` using OpenCV
  - `cv2.VideoCapture(0)` loop
  - Haar cascade face detection (`haarcascade_frontalface_default.xml`)
  - crop face → grayscale → 48×48 → model → emotion label
  - draw bounding box + emotion label on frame
  - smooth predictions over last 5 frames (avoid flickering)
  - press Q to quit
  - reuse `preprocess_image()` from `src/predict.py`
  - Run: `python -m app.realtime --checkpoint outputs/checkpoints/best_<model>.pt --model <name>`

## Training improvements

- [ ] Train SE-CNN with cosine LR scheduler for 50 epochs (not done yet)
- [ ] Try weighted CrossEntropyLoss to help disgust class (547 vs 8989 samples)
- [ ] Two-phase ResNet-18: freeze backbone → warmup head → unfreeze all → fine-tune

## Evaluation gaps

- [ ] Re-evaluate baseline CNN with retrained checkpoint (50 ep + scheduler)
- [ ] Generate confusion matrices and training curves for SE-CNN and ResNet-18
- [ ] Collect all final test numbers into `docs/templates/results_table.csv`

## Report / docs

- [ ] Fill `docs/templates/experiment_log.md` for each training run
- [ ] Write report sections using `docs/templates/report_outline.md`
