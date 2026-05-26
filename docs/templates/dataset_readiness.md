# Dataset readiness record

Use this before claiming baseline metrics. The code can be implemented and smoke-tested with sanitized FER-shaped fixtures, but real baseline training needs the real FER-2013 CSV.

## Dataset identity

- Dataset name: FER-2013
- Expected local path: `data/raw/fer2013.csv`
- Source: Kaggle dataset `msambare/fer2013` (folder format), converted to CSV via `src/data/convert_folders_to_csv.py`
- Download date: 2026-05-26
- License/terms: Kaggle dataset, check Kaggle terms of service
- File checksum command:
  ```bash
  python -c "import hashlib, pathlib; p=pathlib.Path('data/raw/fer2013.csv'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
  ```
- SHA256: `fdcc8d89b81b1ec994a23d8e283df7fdd6df8e04492b58f121ea0dda173dc803`

## Conversion note

The Kaggle `msambare/fer2013` dataset is folder-based (`train/{emotion}/*.jpg`, `test/{emotion}/*.jpg`).
It was converted to CSV using:
```bash
python -m src.data.convert_folders_to_csv --raw-dir data/raw --out data/raw/fer2013.csv
```

**Important:** The `PublicTest` (validation) split is derived from the train folder (10%, seed=42),
NOT from the original Kaggle competition's PublicTest split.
The `test/` folder maps to `PrivateTest`.

## Required schema

| Column | Required | Expected values |
|---|---:|---|
| `emotion` | yes | integers 0-6 |
| `pixels` | yes | 2304 whitespace-separated grayscale values |
| `Usage` | yes | `Training`, `PublicTest`, `PrivateTest` |

## Validation output

- Command:
  ```bash
  python -m src.data.prepare_data --csv data/raw/fer2013.csv --out data/processed/dataset_stats.json
  ```
- Valid rows: 35887
- Removed rows: 0
- Derived stats file: `data/processed/dataset_stats.json`

## Split counts

| Split | Count |
|---|---:|
| Training | 25839 |
| PublicTest | 2870 |
| PrivateTest | 7178 |

## Class counts

| Emotion | Count |
|---|---:|
| anger | 4953 |
| disgust | 547 |
| fear | 5121 |
| happiness | 8989 |
| sadness | 6077 |
| surprise | 4002 |
| neutral | 6198 |

## Pixel statistics (derived)

- mean: 129.38
- std: 65.08

## Class imbalance note

`disgust` has only 547 samples — ~16× smaller than `happiness` (8989).
Use macro-F1 as the primary metric, not accuracy.
Consider weighted loss or oversampling for `disgust` if recall is poor.
