"""Tests for fid-based apply_idx remapping in manual edit diffs.

Verifies that _remap_apply_idx_by_fid correctly resolves fid values to
current positional indices, making diffs immune to global sort-order changes.
"""

import sys
import types
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

# The pipeline module fails to import in test environments due to
# entry_points API differences.  Mock it so we can import diff.py.
_mock_pipeline = types.ModuleType("emeraldprocessing.pipeline")
_mock_pipeline.ProcessingData = MagicMock
sys.modules.setdefault("emeraldprocessing.pipeline", _mock_pipeline)

from emeraldprocessing.diff import _remap_apply_idx_by_fid


def _make_xyz(fids):
    """Create a mock XYZ-like object with flightlines containing fid."""
    fl = pd.DataFrame({"fid": fids})
    xyz = MagicMock()
    xyz.flightlines = fl
    return xyz


def _make_diff(apply_idx, fids=None, inuse_vals=None):
    """Create a mock diff XYZ-like object."""
    fl = pd.DataFrame({"apply_idx": np.array(apply_idx, dtype=np.int64)})
    if fids is not None:
        fl["fid"] = fids
    if inuse_vals is not None:
        fl["InUse_Ch01"] = inuse_vals
    diff = MagicMock()
    diff.flightlines = fl
    return diff


class TestRemapApplyIdxByFid:

    def test_no_fid_in_diff_is_noop(self):
        """Old diffs without fid column should be untouched."""
        target = _make_xyz([100.0, 200.0, 300.0])
        diff = _make_diff([0, 2])

        _remap_apply_idx_by_fid(target, diff)

        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [0, 2])

    def test_basic_remapping(self):
        """Fid-based remapping corrects apply_idx when sort order differs."""
        target = _make_xyz([300.0, 100.0, 200.0])
        diff = _make_diff(apply_idx=[0, 2], fids=[100.0, 300.0])

        _remap_apply_idx_by_fid(target, diff)

        # fid 100.0→pos 1, fid 300.0→pos 0
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [1, 0])
        assert "fid" not in diff.flightlines.columns

    def test_identical_order_is_noop(self):
        """When sort order hasn't changed, remapping produces same indices."""
        target = _make_xyz([100.0, 200.0, 300.0])
        diff = _make_diff(apply_idx=[0, 2], fids=[100.0, 300.0])

        _remap_apply_idx_by_fid(target, diff)

        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [0, 2])

    def test_missing_fid_falls_back(self):
        """If a diff fid is not in the target, fall back to original apply_idx."""
        target = _make_xyz([100.0, 200.0, 300.0])
        diff = _make_diff(apply_idx=[0, 1], fids=[100.0, 999.0])

        _remap_apply_idx_by_fid(target, diff)

        # Fallback: apply_idx unchanged, fid still present
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [0, 1])
        assert "fid" in diff.flightlines.columns

    def test_preserves_other_diff_columns(self):
        """Remapping should not disturb non-fid, non-apply_idx columns."""
        target = _make_xyz([300.0, 100.0, 200.0])
        diff = _make_diff(apply_idx=[0, 1], fids=[100.0, 200.0], inuse_vals=[0, 0])

        _remap_apply_idx_by_fid(target, diff)

        np.testing.assert_array_equal(diff.flightlines["InUse_Ch01"].values, [0, 0])
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [1, 2])

    def test_duplicate_fids_uses_first(self):
        """If target has duplicate fids, the first occurrence is used."""
        target = _make_xyz([100.0, 200.0, 100.0, 300.0])
        diff = _make_diff(apply_idx=[0], fids=[100.0])

        _remap_apply_idx_by_fid(target, diff)

        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [0])

    def test_large_shift_scenario(self):
        """Simulate the actual bug: 54-position shift from NaN date changes."""
        n = 200
        fids = np.arange(1000.0, 1000.0 + n)

        # New order: first 54 elements shifted to later positions
        new_order = np.concatenate([np.arange(54, n), np.arange(0, 54)])
        target = _make_xyz(fids[new_order])

        # Diff targeting positions 60-65 in the old order
        diff_positions = [60, 61, 62, 63, 64, 65]
        diff_fids = fids[diff_positions]
        diff = _make_diff(apply_idx=diff_positions, fids=diff_fids)

        _remap_apply_idx_by_fid(target, diff)

        # Verify: remapped positions point to the same fids in new order
        remapped = diff.flightlines["apply_idx"].values
        for i, orig_pos in enumerate(diff_positions):
            expected_fid = fids[orig_pos]
            actual_fid = fids[new_order[remapped[i]]]
            assert expected_fid == actual_fid
