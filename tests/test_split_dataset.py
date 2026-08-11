from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_curation.split_dataset import split_dataset


def _example(i: int, difficulty: str = "easy") -> dict:
    return {
        "id": f"ex-{i:03d}",
        "scenario_id": "unit_test",
        "difficulty": difficulty,
        "document_text": f"Invoice {i}\nTotal: ${i + 1}.00",
        "ground_truth_json": {
            "vendor_name": "Test Vendor",
            "line_items": [{"description": "Item", "total": float(i + 1)}],
            "total_amount": float(i + 1),
            "currency": "USD",
        },
    }


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def test_split_dataset_preserves_eval_minimum_for_small_valid_dataset(tmp_path: Path):
    input_path = tmp_path / "quality_filtered.jsonl"
    examples = [_example(i, ["easy", "medium", "hard"][i % 3]) for i in range(52)]
    _write_jsonl(input_path, examples)

    stats = split_dataset(
        input_path=input_path,
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        eval_path=tmp_path / "eval_locked.jsonl",
        eval_fraction=0.12,
        eval_min=50,
        seed=123,
    )

    assert stats["n_eval"] == 50
    assert stats["n_val"] == 1
    assert stats["n_train"] == 1
    assert (tmp_path / "eval_locked.jsonl.sha256").exists()


def test_split_dataset_rejects_when_no_room_for_train_and_val(tmp_path: Path):
    input_path = tmp_path / "quality_filtered.jsonl"
    _write_jsonl(input_path, [_example(i) for i in range(51)])

    with pytest.raises(ValueError, match="leaving at least one validation"):
        split_dataset(
            input_path=input_path,
            train_path=tmp_path / "train.jsonl",
            val_path=tmp_path / "val.jsonl",
            eval_path=tmp_path / "eval_locked.jsonl",
            eval_min=50,
        )

