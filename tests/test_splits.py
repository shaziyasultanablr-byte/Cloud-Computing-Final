"""Dataset manifest and cross-load split disjointness tests for VibeGuard Cloud."""

from __future__ import annotations

import copy
import pytest

from data_utils import DataValidationError, load_dataset_manifest, verify_split_disjointness


def test_manifest_has_16_disjoint_recordings() -> None:
    rows = load_dataset_manifest("docs/DATASET_MANIFEST.csv")
    assert len(rows) == 16

    expected_ids = {97, 98, 99, 100, 105, 106, 107, 108, 118, 119, 120, 121, 130, 131, 132, 133}
    actual_ids = {r["file_id"] for r in rows}
    assert actual_ids == expected_ids

    train_loads = {r["load_hp"] for r in rows if r["split"] == "train"}
    val_loads = {r["load_hp"] for r in rows if r["split"] == "val"}
    test_loads = {r["load_hp"] for r in rows if r["split"] == "test"}

    assert train_loads == {0, 1}
    assert val_loads == {2}
    assert test_loads == {3}


def test_split_validator_rejects_load_leakage() -> None:
    rows = load_dataset_manifest("docs/DATASET_MANIFEST.csv")
    corrupted = copy.deepcopy(rows)
    # Leak a 3 HP recording into training
    for r in corrupted:
        if r["file_id"] == 98:
            r["load_hp"] = 3
            break

    with pytest.raises(DataValidationError, match="Load leakage detected"):
        verify_split_disjointness(corrupted)


def test_split_validator_rejects_duplicate_file_ids() -> None:
    rows = load_dataset_manifest("docs/DATASET_MANIFEST.csv")
    corrupted = copy.deepcopy(rows)
    corrupted[1]["file_id"] = corrupted[0]["file_id"]

    with pytest.raises(DataValidationError, match="Duplicate file_id"):
        verify_split_disjointness(corrupted)
