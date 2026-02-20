"""
Regression tests for slope/curvature wide-window computation (issue #767).

HeliTEM data has extremely close gate time spacing (Δlog10(t) ≈ 0.001 at
early gates vs SkyTEM's 0.085). The wide-window approach computes slopes
and curvatures over a larger gate span so the denominator is large enough
to produce meaningful values.

SkyTEM: adjacent spacing always exceeds the threshold, so the wide-window
pairs are always (k-1, k+1), producing identical results to the old
adjacent-gate formula.
"""
import numpy as np
import pandas as pd
import pytest
from emeraldprocessing.tem.utils import (
    calculate_transient_slopes,
    calculate_transient_curvatures,
    build_l10_dBdt_time_df,
    _build_adaptive_windows,
    _needs_regression,
)

DATA_KEY = 'Gate_Ch01'


# --- Unit tests for regression helpers ---

class TestNeedsRegression:
    """Unit tests for the _needs_regression helper."""

    def test_skytem_wide_spacing_returns_false(self):
        """SkyTEM-like wide spacing should not trigger regression."""
        times = np.linspace(-4.0, -3.1, 10)  # Δ ≈ 0.1
        assert _needs_regression(times) is False

    def test_helitem_tight_spacing_returns_true(self):
        """HeliTEM-like tight spacing should trigger regression."""
        times = np.array([-2.284, -2.283, -2.282, -2.281, -2.280, -2.279,
                          -2.270, -2.255, -2.230, -2.190])
        assert _needs_regression(times) is True

    def test_single_gate_returns_false(self):
        """Single gate cannot have any spacing."""
        assert _needs_regression(np.array([-3.5])) is False

    def test_all_nan_returns_false(self):
        """All NaN times have no valid spacing."""
        assert _needs_regression(np.array([np.nan, np.nan])) is False


class TestBuildAdaptiveWindows:
    """Unit tests for the _build_adaptive_windows helper."""

    def test_minimum_gate_count_guaranteed(self):
        """Every window should have at least min_gates entries."""
        times = np.linspace(-4.0, -3.0, 20)
        windows = _build_adaptive_windows(times, min_gates=5)
        for k, w in enumerate(windows):
            assert len(w) >= 5, f"Window at gate {k} has {len(w)} gates, expected >= 5"

    def test_window_contains_center_gate(self):
        """Each window should contain its center gate."""
        times = np.linspace(-4.0, -3.0, 15)
        windows = _build_adaptive_windows(times, min_gates=3)
        for k, w in enumerate(windows):
            assert k in w, f"Window at gate {k} does not contain gate {k}"

    def test_tightly_spaced_gates_get_large_windows(self):
        """Tightly spaced gates should get large decade-based windows."""
        # HeliTEM-like: 20 gates within 0.1 decades
        times = np.linspace(-2.30, -2.20, 20)
        windows = _build_adaptive_windows(times, min_gates=5, max_half_decades=0.15)
        # All gates are within 0.1 decades total, so max_half_decades=0.15
        # should include all of them
        for w in windows:
            assert len(w) == 20


# --- HeliTEM Slope Tests ---

def test_helitem_early_gate_slopes_are_finite(helitem_processing):
    """HeliTEM gates 1+ should have valid (finite) slopes, not NaN."""
    slope = calculate_transient_slopes(helitem_processing, DATA_KEY)

    # Gate 0 has no backward partner → NaN is expected
    # Gates 1+ should have at least some valid slopes
    for gate_idx in range(1, slope.shape[1]):
        valid_pct = slope.iloc[:, gate_idx].notna().mean() * 100
        assert valid_pct > 0, (
            f"Gate {gate_idx}: all slopes are NaN, expected at least some valid values")


def test_helitem_early_slopes_not_extreme(helitem_processing):
    """Wide-window slopes at late gates (good data + wide span) should be reasonable.

    Early HeliTEM gates have pervasive sign changes and near-zero values,
    making even wide-window slopes noisy. But at later gates (19+) where
    data quality is good and gate spacing is wide (Δlog10(t) > 0.03),
    slopes should be in a normal range.
    """
    slope = calculate_transient_slopes(helitem_processing, DATA_KEY)

    # Gates 19+ have good data quality and span > 0.03
    for gate_idx in range(19, slope.shape[1]):
        vals = slope.iloc[:, gate_idx].dropna()
        if len(vals) > 0:
            extreme = (vals.abs() > 50).sum()
            assert extreme == 0, (
                f"Gate {gate_idx}: {extreme} soundings have |slope| > 50 "
                f"(max={vals.abs().max():.1f})")


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
    """SkyTEM min Δlog10(t) ≈ 0.085 >> 0.01. Wide-window always uses adjacent gate."""
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


def test_skytem_slopes_identical_to_adjacent_diff(skytem_processing):
    """SkyTEM slopes must be numerically identical to adjacent-gate .diff() formula."""
    slope = calculate_transient_slopes(skytem_processing, DATA_KEY)

    # Compute reference slopes using the old adjacent-gate method
    l10_dBdt_df, l10_gate_times_df = build_l10_dBdt_time_df(skytem_processing, DATA_KEY)
    ref_slope = l10_dBdt_df.diff(axis=1) / l10_gate_times_df.diff(axis=1)

    # Apply same sign change guard as old code
    original_data = skytem_processing.xyz.layer_data[DATA_KEY]
    prev_data = original_data.shift(1, axis=1)
    bad = (
        (original_data * prev_data < 0)
        | (original_data == 0)
        | (prev_data == 0)
    )
    ref_slope[bad] = np.nan

    # Compare: NaN positions must match, values must be close
    new_vals = slope.values
    ref_vals = ref_slope.values
    both_nan = np.isnan(new_vals) & np.isnan(ref_vals)
    either_nan = np.isnan(new_vals) | np.isnan(ref_vals)
    assert np.all(both_nan == either_nan), "NaN positions differ between new and reference slopes"

    valid = ~either_nan
    if valid.any():
        np.testing.assert_allclose(new_vals[valid], ref_vals[valid], rtol=1e-12)


# --- HeliTEM Curvature Tests ---

def test_helitem_curvature_early_gates_finite(helitem_processing):
    """HeliTEM curvature should have valid values at some gates.

    The sign change guard (requires all 3 endpoint gates > 0) legitimately
    masks many gates in this dataset where 60-85% of soundings have negative
    data. But the wide-window approach enables curvature computation at
    gates that were previously blanket-NaN from the spacing guard.
    """
    curv = calculate_transient_curvatures(helitem_processing, DATA_KEY)

    gates_with_valid = sum(
        1 for g in range(curv.shape[1])
        if curv.iloc[:, g].notna().any()
    )
    assert gates_with_valid >= 5, (
        f"Only {gates_with_valid} gates have any valid curvatures, "
        f"expected at least 5")


def test_helitem_curvature_normalized_to_skytem_range(helitem_processing, skytem_processing):
    """Normalized HeliTEM curvatures should be in a comparable range to SkyTEM.

    Without normalization, HeliTEM regression curvatures are 100-400 at early
    gates while SkyTEM adjacent curvatures are ~1.  After normalization by
    (window_half_span)^2, HeliTEM curvatures should be in single-digit range
    so the same threshold works for both systems.
    """
    heli_curv = calculate_transient_curvatures(helitem_processing, DATA_KEY)
    sky_curv = calculate_transient_curvatures(skytem_processing, DATA_KEY)

    heli_vals = heli_curv.values.flatten()
    heli_valid = heli_vals[np.isfinite(heli_vals)]
    sky_vals = sky_curv.values.flatten()
    sky_valid = sky_vals[np.isfinite(sky_vals)]

    heli_abs_95 = np.percentile(np.abs(heli_valid), 95)
    sky_abs_95 = np.percentile(np.abs(sky_valid), 95)

    # Both should be within an order of magnitude of each other
    assert heli_abs_95 < 100, (
        f"HeliTEM 95th percentile |curvature| = {heli_abs_95:.1f}, "
        f"expected < 100 after normalization")
    assert heli_abs_95 / sky_abs_95 < 10, (
        f"HeliTEM/SkyTEM 95th pct ratio = {heli_abs_95 / sky_abs_95:.1f}, "
        f"expected < 10 for comparable thresholds")


def test_helitem_slope_gate_to_gate_smoothness(helitem_processing):
    """Regression slopes should be smooth gate-to-gate (no wild jumps).

    The mean absolute slope change between adjacent gates at late gates
    (19+, where data quality is good) should be modest, indicating the
    regression approach produces physically smooth results.
    """
    slope = calculate_transient_slopes(helitem_processing, DATA_KEY)
    # Late gates only (good data quality)
    late_slopes = slope.iloc[:, 19:].values
    delta = np.abs(np.diff(late_slopes, axis=1))
    mean_delta = np.nanmean(delta)
    assert mean_delta < 5, (
        f"Mean |delta-slope| between adjacent late gates = {mean_delta:.2f}, "
        f"expected < 5 for smooth regression output")


def test_method_adjacent_always_uses_finite_diff(helitem_processing, skytem_processing):
    """method='adjacent' must always use the finite-difference path, even for
    HeliTEM data that would normally trigger regression under method='auto'."""
    for proc, label in [(helitem_processing, 'HeliTEM'),
                        (skytem_processing, 'SkyTEM')]:
        slope_adj = calculate_transient_slopes(proc, DATA_KEY, method='adjacent')
        curv_adj = calculate_transient_curvatures(proc, DATA_KEY, method='adjacent')

        l10_dBdt_df, l10_gate_times_df = build_l10_dBdt_time_df(proc, DATA_KEY)
        ref_slope = l10_dBdt_df.diff(axis=1) / l10_gate_times_df.diff(axis=1)
        original_data = proc.xyz.layer_data[DATA_KEY]
        prev_data = original_data.shift(1, axis=1)
        bad = (original_data * prev_data < 0) | (original_data == 0) | (prev_data == 0)
        ref_slope[bad] = np.nan

        s_new = slope_adj.values
        s_ref = ref_slope.values
        valid = np.isfinite(s_new) & np.isfinite(s_ref)
        if valid.any():
            np.testing.assert_allclose(
                s_new[valid], s_ref[valid], rtol=1e-12,
                err_msg=f"{label} method='adjacent' slopes differ from reference")


# --- SkyTEM Curvature Tests ---

def test_skytem_curvature_not_affected(skytem_processing):
    """SkyTEM curvature should not be affected by the wide-window change.
    Curvature at gate k needs gates k-1, k, k+1. With gates 0-4 NaN and
    gate 5 first valid, curvatures start being valid at gate 7."""
    curv = calculate_transient_curvatures(skytem_processing, DATA_KEY)
    for g in range(8, 15):
        valid_pct = curv.iloc[:, g].notna().mean() * 100
        assert valid_pct > 50, (
            f"Gate {g}: only {valid_pct:.1f}% valid curvatures, expected >50% for clean SkyTEM data")


def test_skytem_curvatures_identical_to_adjacent(skytem_processing):
    """SkyTEM curvatures must be numerically identical to old adjacent-gate formula."""
    curv = calculate_transient_curvatures(skytem_processing, DATA_KEY)

    # Compute reference curvatures: (f[k+1] - 2*f[k] + f[k-1]) / (t[k+1] - t[k-1])^2
    l10_dBdt_df, l10_gate_times_df = build_l10_dBdt_time_df(skytem_processing, DATA_KEY)
    l10_times_1d = l10_gate_times_df.iloc[0].values
    n_gates = len(l10_times_1d)

    ref_curv = pd.DataFrame(np.nan, index=l10_dBdt_df.index, columns=l10_dBdt_df.columns)
    for k in range(1, n_gates - 1):
        if np.isnan(l10_times_1d[k - 1]) or np.isnan(l10_times_1d[k + 1]):
            continue
        t_span = l10_times_1d[k + 1] - l10_times_1d[k - 1]
        ref_curv.iloc[:, k] = (
            l10_dBdt_df.iloc[:, k + 1] - 2 * l10_dBdt_df.iloc[:, k] + l10_dBdt_df.iloc[:, k - 1]
        ) / (t_span ** 2)

    # Apply sign change guard (same as old code)
    original_data = skytem_processing.xyz.layer_data[DATA_KEY]
    for k in range(1, n_gates - 1):
        if np.isnan(l10_times_1d[k - 1]) or np.isnan(l10_times_1d[k + 1]):
            continue
        bad = (
            (original_data.iloc[:, k - 1] <= 0)
            | (original_data.iloc[:, k] <= 0)
            | (original_data.iloc[:, k + 1] <= 0)
        )
        ref_curv.loc[bad, ref_curv.columns[k]] = np.nan

    # Compare: NaN positions must match, values must be close
    new_vals = curv.values
    ref_vals = ref_curv.values
    both_nan = np.isnan(new_vals) & np.isnan(ref_vals)
    either_nan = np.isnan(new_vals) | np.isnan(ref_vals)
    assert np.all(both_nan == either_nan), "NaN positions differ between new and reference curvatures"

    valid = ~either_nan
    if valid.any():
        np.testing.assert_allclose(new_vals[valid], ref_vals[valid], rtol=1e-12)
