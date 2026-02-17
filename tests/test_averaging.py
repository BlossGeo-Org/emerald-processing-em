"""
Tests for hybrid averaging functions.

Key test: Verifies that NaN values (culled data) don't cause NaN STD
in neighboring positions - the core bug fix from SPRINT_03.
"""
import numpy as np
import pandas as pd
import pytest
from emeraldprocessing.tem.utils import (
    alpha_trim,
    get_min_periods,
    inverse_variance_weights,
    rolling_hybrid_mean_df
)


def test_get_min_periods_percentage():
    """Test percentage-based min_periods calculation."""
    assert get_min_periods(10, min_fraction=0.35) == 4
    assert get_min_periods(10) == 4  # Default 35%
    assert get_min_periods(3, min_fraction=0.35) == 2  # Minimum of 2


def test_hybrid_handles_nan_near_culled():
    """
    Core bug fix test: NaN values (culled data) should not cause
    NaN output in neighboring positions.

    Before fix: positions adjacent to culled data got NaN STD
    After fix: positions adjacent to culled data get valid STD
    """
    # Create data with a gap (NaN) in the middle - simulates culled data
    data = pd.DataFrame({'col': [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0]})
    errors = pd.DataFrame({'col': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]})
    rolling_lengths = [3]

    ave, err = rolling_hybrid_mean_df(data, errors, rolling_lengths,
                                      alpha=0.0, min_fraction=0.3)

    # Positions 2 and 4 (adjacent to NaN at position 3) should NOT be NaN
    assert not np.isnan(ave.loc[2, 'col']), "Position adjacent to culled should have valid mean"
    assert not np.isnan(ave.loc[4, 'col']), "Position adjacent to culled should have valid mean"
    assert not np.isnan(err.loc[2, 'col']), "Position adjacent to culled should have valid STD"
    assert not np.isnan(err.loc[4, 'col']), "Position adjacent to culled should have valid STD"


def test_alpha_trim_small_window():
    """
    Regression test for issue #766: alpha_trim must not destroy data
    when the rolling window is small (e.g. filter_length=3).

    Bug: max(1, int(n * alpha)) forced n_trim=1 even for n=3,
    trimming 2 of 3 values, leaving only 1 which failed the
    variance check (needs >= 2). This wiped out all early gates
    when width_at_first_gate=3.
    """
    data = np.array([107.1, 106.0, 55.5])
    errors = np.array([0.1071, 0.1060, 0.0555])

    trimmed_data, trimmed_err = alpha_trim(data, errors, alpha=0.1)

    # With 3 samples and alpha=0.1, int(3*0.1)=0 so no trimming should occur
    assert len(trimmed_data) == 3, (
        f"alpha_trim should not trim a window of 3 (got {len(trimmed_data)})"
    )


def test_hybrid_small_filter_length_produces_output():
    """
    Regression test for issue #766: filter_length=3 with default alpha=0.1
    must produce valid output for all soundings, not just line edges.

    This simulates the HeliTEM scenario: 26 soundings, single channel,
    all data populated. Early gates (filter_length=3) must not be NaN.
    """
    n_soundings = 26
    data = pd.DataFrame({'gate0': np.linspace(50, 110, n_soundings),
                          'gate24': np.linspace(0.5, 2.5, n_soundings)})
    errors = pd.DataFrame({'gate0': [0.001] * n_soundings,
                            'gate24': [0.001] * n_soundings})
    rolling_lengths = [3, 5]  # first gate=3, last gate=5

    ave, err = rolling_hybrid_mean_df(data, errors, rolling_lengths)

    # All interior soundings (not just edges) must have valid output
    mid = n_soundings // 2
    assert not np.isnan(ave.loc[mid, 'gate0']), (
        "Early gate (filter_length=3) should produce valid mean at mid-line"
    )
    assert not np.isnan(err.loc[mid, 'gate0']), (
        "Early gate (filter_length=3) should produce valid STD at mid-line"
    )
    # Count how many soundings got valid output for early gate
    valid_count = ave['gate0'].notna().sum()
    assert valid_count == n_soundings, (
        f"All {n_soundings} soundings should have valid early gate data, got {valid_count}"
    )


def test_inverse_variance_weights_zero_error():
    """
    Regression test for issue #766 follow-up: near-zero data values with
    fractional error model produce zero absolute errors, causing infinite
    IVW weights and NaN/10^-18 artifacts in the weighted mean.

    The error floor prevents zero-error values from dominating.
    """
    # Late gate scenario: one value is exactly 0.0, others are small
    errors = np.array([0.0, 0.0002, 0.0009])
    w = inverse_variance_weights(errors)

    assert not np.any(np.isinf(w)), "Weights must not be infinite"
    assert not np.any(np.isnan(w)), "Weights must not be NaN"


def test_hybrid_near_zero_data_no_artifacts():
    """
    Regression test for issue #766 follow-up: averaging late gate data
    with values near zero must not produce tiny artifacts (10^-18).

    With fractional errors, value=0.0 gets abs_err=0.0, which without
    a floor gives infinite weight, pulling the average to ~0 or NaN.
    """
    n = 10
    # Late gate data with a zero value and noise-floor values
    data = pd.DataFrame({'col': [0.0, 0.2, -0.9, -6.6, -1.2,
                                  -2.8, -4.0, -3.7, -4.7, -3.3]})
    errors = pd.DataFrame({'col': [0.001] * n})
    rolling_lengths = [5]

    ave, err = rolling_hybrid_mean_df(data, errors, rolling_lengths)

    # No averaged value should be smaller than ~0.01 in magnitude
    # (the raw data ranges from -6.6 to 0.2, so averages should be in that range)
    for i in range(n):
        val = ave.loc[i, 'col']
        if not np.isnan(val):
            assert abs(val) > 1e-10, (
                f"Sounding {i}: averaged value {val:.2e} is suspiciously tiny"
            )
