"""
Regression tests for slope/curvature guard conditions (issue #767).

HeliTEM data has extremely close gate time spacing and frequent sign changes,
causing slope/curvature filters to disable gates uniformly across all soundings.
The fix adds two guards that mask unreliable computations with NaN.
"""
import numpy as np
import pytest
from emeraldprocessing.tem.utils import calculate_transient_slopes, calculate_transient_curvatures

DATA_KEY = 'Gate_Ch01'

# --- HeliTEM Slope Tests ---

def test_helitem_spacing_guard_masks_early_gates(helitem_processing):
    """HeliTEM gates with Δlog10(t) < 0.01 must have slopes set to NaN."""
    slope = calculate_transient_slopes(helitem_processing, DATA_KEY)
    gt = helitem_processing.GateTimes[DATA_KEY]
    dlog = np.diff(np.log10(gt[gt > 0]))

    # Find which gates have narrow spacing
    narrow_count = sum(1 for d in dlog if d < 0.01)
    assert narrow_count > 0, "HeliTEM should have narrow-spaced gates"

    # Verify those gates are masked
    for gate_idx in range(1, len(dlog) + 1):
        if dlog[gate_idx - 1] < 0.01:
            assert slope.iloc[:, gate_idx].isna().all(), (
                f"Gate {gate_idx}: slope should be NaN (spacing guard)")


def test_helitem_later_gates_have_valid_slopes(helitem_processing):
    """HeliTEM gates with adequate spacing should have at least some valid slopes."""
    slope = calculate_transient_slopes(helitem_processing, DATA_KEY)
    gt = helitem_processing.GateTimes[DATA_KEY]
    dlog = np.diff(np.log10(gt[gt > 0]))

    for gate_idx in range(1, len(dlog) + 1):
        if dlog[gate_idx - 1] >= 0.01:
            valid_pct = slope.iloc[:, gate_idx].notna().mean() * 100
            assert valid_pct > 0, f"Gate {gate_idx}: no valid slopes at all"


def test_helitem_slopes_not_uniform_at_later_gates(helitem_processing):
    """After fix, later-gate slopes should vary per sounding (not uniform disabling)."""
    slope = calculate_transient_slopes(helitem_processing, DATA_KEY)
    gt = helitem_processing.GateTimes[DATA_KEY]
    dlog = np.diff(np.log10(gt[gt > 0]))

    # Find well-spaced gates
    well_spaced = [i + 1 for i, d in enumerate(dlog) if d >= 0.01]
    for gate_idx in well_spaced[:3]:  # Check first few well-spaced gates
        vals = slope.iloc[:, gate_idx].dropna()
        if len(vals) > 10:
            assert vals.std() > 0.1, (
                f"Gate {gate_idx}: slopes suspiciously uniform (std={vals.std():.4f})")


# --- SkyTEM Slope Tests (backward compatibility) ---

def test_skytem_no_gates_masked_by_spacing_guard(skytem_processing):
    """SkyTEM min Δlog10(t) ≈ 0.085 >> 0.01. Spacing guard must not affect SkyTEM."""
    gt = skytem_processing.GateTimes[DATA_KEY]
    dlog = np.diff(np.log10(gt[gt > 0]))
    assert all(d >= 0.01 for d in dlog), (
        f"SkyTEM gate spacing should all be >= 0.01, min={min(dlog):.4f}")


def test_skytem_early_gate_slopes_reasonable(skytem_processing):
    """SkyTEM early-gate slopes should be in [-5, 0] range for clean data."""
    slope = calculate_transient_slopes(skytem_processing, DATA_KEY)
    # Check a mid-range gate (gate 10) — well-spaced, clean data
    vals = slope.iloc[:, 10].dropna()
    assert len(vals) > 100, "Should have many valid slopes at gate 10"
    assert -5 < vals.mean() < 0, (
        f"Mean slope at gate 10 should be in [-5, 0], got {vals.mean():.2f}")
    # No sounding should have extreme slopes at clean gates
    assert (vals.abs() > 20).sum() == 0, "No extreme slopes expected at clean SkyTEM gates"


def test_skytem_slopes_unchanged_at_clean_gates(skytem_processing):
    """SkyTEM slopes at clean gates (with data) should have valid data for most soundings.
    Gates 0-4 are NaN in this dataset; gate 5 is the first with data, so slopes
    start being valid at gate 6 (needs both current and previous gate)."""
    slope = calculate_transient_slopes(skytem_processing, DATA_KEY)
    for g in range(7, 16):
        valid_pct = slope.iloc[:, g].notna().mean() * 100
        assert valid_pct > 50, (
            f"Gate {g}: only {valid_pct:.1f}% valid slopes, expected >50% for clean SkyTEM data")


# --- HeliTEM Curvature Tests ---

def test_helitem_curvature_spacing_guard(helitem_processing):
    """HeliTEM curvature should be NaN at closely-spaced gates."""
    curv = calculate_transient_curvatures(helitem_processing, DATA_KEY)
    gt = helitem_processing.GateTimes[DATA_KEY]
    dlog = np.diff(np.log10(gt[gt > 0]))

    # Curvature at gate k uses gates k-1, k, k+1 — if either adjacent
    # spacing is narrow, the curvature should be NaN
    for col_idx in range(1, len(curv.columns) - 1):
        if col_idx - 1 < len(dlog) and col_idx < len(dlog):
            if dlog[col_idx - 1] < 0.01 or dlog[col_idx] < 0.01:
                assert curv.iloc[:, col_idx].isna().all(), (
                    f"Curvature at gate {col_idx}: should be NaN (spacing guard)")


# --- SkyTEM Curvature Tests ---

def test_skytem_curvature_not_affected(skytem_processing):
    """SkyTEM curvature should not be affected by the spacing guard.
    Curvature at gate k needs gates k-1, k, k+1. With gates 0-4 NaN and
    gate 5 first valid, curvatures start being valid at gate 7."""
    curv = calculate_transient_curvatures(skytem_processing, DATA_KEY)
    for g in range(8, 15):
        valid_pct = curv.iloc[:, g].notna().mean() * 100
        assert valid_pct > 50, (
            f"Gate {g}: only {valid_pct:.1f}% valid curvatures, expected >50% for clean SkyTEM data")
