from src.data.prepare_data import validate_fer2013_csv


def test_validate_fer2013_csv_counts_valid_and_invalid_rows(sample_fer_csv):
    stats = validate_fer2013_csv(sample_fer_csv)

    assert stats["valid_rows"] == 3
    assert stats["removed_rows"] == 1
    assert stats["split_counts"] == {"Training": 1, "PublicTest": 1, "PrivateTest": 1}
    assert stats["class_counts"]["anger"] == 1
    assert stats["class_counts"]["happiness"] == 1
    assert stats["class_counts"]["neutral"] == 1
    assert 0 <= stats["pixel_mean"] <= 255
    assert stats["pixel_std"] >= 0


def test_validate_fer2013_csv_rejects_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("emotion,pixels\n0,1 2 3\n", encoding="utf-8")

    try:
        validate_fer2013_csv(bad_csv)
    except ValueError as exc:
        assert "Usage" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing Usage column")
