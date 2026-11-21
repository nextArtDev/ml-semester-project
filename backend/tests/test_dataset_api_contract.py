"""Contract tests for dataset summary endpoints without touching the database."""

from __future__ import annotations

from datetime import datetime, timezone

from app.api import datasets as api_datasets
from app.api.schemas import DatasetSummary


def _row() -> dict:
    return {
        "id": 116,
        "name": "Forest CoverType",
        "source": "covertype",
        "n_rows": 53000,
        "n_features": 54,
        "n_classes": 7,
        "dimension": 54,
        "class_labels": [0, 1, 2, 3, 4, 5, 6],
        "preprocessing_mode": "covertype",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }


def test_list_datasets_selects_and_returns_preprocessing_mode(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_fetch_all(sql: str, params=None):
        captured["sql"] = sql
        return [_row()]

    monkeypatch.setattr(api_datasets, "fetch_all", fake_fetch_all)

    result = api_datasets.list_datasets()

    assert "preprocessing_mode" in captured["sql"]
    assert len(result) == 1
    assert isinstance(result[0], DatasetSummary)
    assert result[0].preprocessing_mode == "covertype"


def test_get_dataset_selects_and_returns_preprocessing_mode(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_fetch_one(sql: str, params=None):
        captured["sql"] = sql
        return _row()

    monkeypatch.setattr(api_datasets, "fetch_one", fake_fetch_one)

    result = api_datasets.get_dataset(116)

    assert "preprocessing_mode" in captured["sql"]
    assert isinstance(result, DatasetSummary)
    assert result.preprocessing_mode == "covertype"


def test_get_dataset_still_reports_missing_dataset(monkeypatch) -> None:
    monkeypatch.setattr(api_datasets, "fetch_one", lambda sql, params=None: None)

    try:
        api_datasets.get_dataset(999)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404
    else:
        raise AssertionError("missing dataset should raise HTTPException 404")
