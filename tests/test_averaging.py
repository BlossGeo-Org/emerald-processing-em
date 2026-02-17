"""
Tests for hybrid averaging functions.

Key test: Verifies that NaN values (culled data) don't cause NaN STD
in neighboring positions - the core bug fix from SPRINT_03.
"""
import numpy as np
import pandas as pd
import pytest
from emeraldprocessing.tem.utils import (
    get_min_periods,
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
