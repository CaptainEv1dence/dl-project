from pathlib import Path

from src.train import run_training


def test_run_training_fast_dev_mode_creates_checkpoint(sample_fer_csv, tmp_path: Path):
    result = run_training(
        csv_path=sample_fer_csv,
        model_name="baseline_cnn",
        output_dir=tmp_path,
        epochs=1,
        batch_size=2,
        fast_dev_run=True,
    )
    assert result["best_checkpoint"].exists()
    assert "best_val_macro_f1" in result
