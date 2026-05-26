from pathlib import Path

from src.evaluate import evaluate_checkpoint
from src.train import run_training


def test_evaluate_checkpoint_returns_metrics(sample_fer_csv, tmp_path: Path):
    trained = run_training(
        csv_path=sample_fer_csv,
        model_name="baseline_cnn",
        output_dir=tmp_path,
        epochs=1,
        batch_size=2,
        fast_dev_run=True,
    )

    metrics = evaluate_checkpoint(
        csv_path=sample_fer_csv,
        checkpoint_path=trained["best_checkpoint"],
        model_name="baseline_cnn",
        split="test",
        output_dir=tmp_path,
    )

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
