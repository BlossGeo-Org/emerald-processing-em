"""
Tests for width-1 moving-average windows.

Issue #31: a filter width of 1 is the natural way to express "decimate but do
not stack". Before the fix the hybrid method returned an all-NaN dataset for
width 1, and the simple method returned all-NaN errors, in both cases silently.
"""
import numpy as np
import pandas as pd

from emeraldprocessing.tem.utils import (
    rolling_hybrid_mean_df,
    rolling_SST_mean_df,
    rolling_mean_df,
    warn_on_new_all_nan_gates,
)


def _data():
    return pd.DataFrame({'g0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})


def _errs():
    return pd.DataFrame({'g0': [0.10, 0.20, 0.05, 0.10, 0.30, 0.10, 0.25]})


def test_hybrid_width_one_is_identity():
    """Width 1 must pass data and its own error through untouched."""
    data, errs = _data(), _errs()
    ave, frac = rolling_hybrid_mean_df(data, errs, [1])

    assert not ave['g0'].isna().any(), "width-1 hybrid returned NaN data"
    np.testing.assert_allclose(ave['g0'].values, data['g0'].values)
    np.testing.assert_allclose(frac['g0'].values, errs['g0'].values)


def test_sst_width_one_is_identity():
    data, errs = _data(), _errs()
    ave, frac = rolling_SST_mean_df(data, errs, [1])

    assert not ave['g0'].isna().any()
    np.testing.assert_allclose(ave['g0'].values, data['g0'].values)
    np.testing.assert_allclose(frac['g0'].values, errs['g0'].values)


def test_simple_width_one_is_identity_when_errors_supplied():
    data, errs = _data(), _errs()
    ave, frac = rolling_mean_df(data, [1], error_calc_scheme='STD', df_err_fp=errs)

    assert not ave['g0'].isna().any()
    np.testing.assert_allclose(ave['g0'].values, data['g0'].values)
    np.testing.assert_allclose(frac['g0'].values, errs['g0'].values)


def test_simple_width_one_without_errors_keeps_data():
    """No error input means no derivable error, but the data must survive."""
    data = _data()
    ave, frac = rolling_mean_df(data, [1], error_calc_scheme='STD')

    np.testing.assert_allclose(ave['g0'].values, data['g0'].values)
    assert frac['g0'].isna().all(), "no error is derivable from a one-sample window"


def test_width_one_preserves_nan_positions():
    """Culled soundings stay culled; width 1 must not invent values."""
    data = pd.DataFrame({'g0': [1.0, np.nan, 3.0, 4.0]})
    errs = pd.DataFrame({'g0': [0.1, np.nan, 0.1, 0.1]})

    ave, frac = rolling_hybrid_mean_df(data, errs, [1])
    assert np.isnan(ave['g0'].iloc[1])
    np.testing.assert_allclose(ave['g0'].iloc[[0, 2, 3]].values, [1.0, 3.0, 4.0])


def test_mixed_widths_only_identity_where_width_is_one():
    """A trapeze filter may be width 1 at one gate and wider at another."""
    data = pd.DataFrame({'g0': [1.0, 2.0, 3.0, 4.0, 5.0],
                         'g1': [1.0, 2.0, 3.0, 4.0, 5.0]})
    errs = pd.DataFrame({'g0': [0.1] * 5, 'g1': [0.1] * 5})

    ave, _ = rolling_hybrid_mean_df(data, errs, [1, 3])
    np.testing.assert_allclose(ave['g0'].values, data['g0'].values)
    # The width-3 gate is genuinely averaged, so its interior differs from
    # the raw data at the ends but is finite everywhere.
    assert not ave['g1'].isna().any()


def test_warn_on_new_all_nan_gates_detects_emptied_gate(capsys):
    before = pd.DataFrame({'g0': [1.0, 2.0], 'g1': [1.0, 2.0]})
    after = pd.DataFrame({'g0': [1.0, 2.0], 'g1': [np.nan, np.nan]})

    emptied = warn_on_new_all_nan_gates(before, after, 'Gate_Ch01')

    assert emptied == ['g1']
    assert 'entirely NaN' in capsys.readouterr().out


def test_warn_on_new_all_nan_gates_ignores_already_empty(capsys):
    """A gate that was already empty going in is not newly broken."""
    before = pd.DataFrame({'g0': [np.nan, np.nan]})
    after = pd.DataFrame({'g0': [np.nan, np.nan]})

    assert warn_on_new_all_nan_gates(before, after, 'Gate_Ch01') == []
    assert capsys.readouterr().out == ''
