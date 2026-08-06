"""
Tests for gate-time lookup across both GEX conventions.

GEX files store gate times either as one shared General/GateTime table that
every channel slices, or as per-moment General/GateTimeLM and General/GateTimeHM
tables. Systems whose moments have independent gate layouts — SkyTEM 306HP has
25 LM gates and 41 HM gates — can only use the second form.

Before the fix, getGateTimesFromGEX hardcoded 'GateTime' and raised a bare
KeyError on any per-moment file, taking out the whole process_tem pipeline for
those systems.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from emeraldprocessing.tem.utils import getGateTimeTable, getGateTimesFromGEX


def _gex(general, channels):
    """Minimal stand-in for libaarhusxyz.GEX — only gex_dict is read."""
    g = SimpleNamespace()
    g.gex_dict = dict(general=general, **channels)
    g.gex_dict['General'] = g.gex_dict.pop('general')
    return g


def _table(n, start=1e-6, step=1e-6):
    """(n, 3) centre/open/close table, the shape GEX gate tables come in."""
    centre = start + step * np.arange(n)
    return np.column_stack([centre, centre - step / 2, centre + step / 2])


# --- shared-table convention (SkyTEM 304, 304M) --------------------------------

def test_shared_table_used_when_present():
    gex = _gex(
        {'GateTime': _table(37)},
        {'Channel1': {'NoGates': 28, 'TransmitterMoment': 'LM',
                      'GateTimeShift': 0.0, 'MeaTimeDelay': 0.0},
         'Channel2': {'NoGates': 37, 'TransmitterMoment': 'HM',
                      'GateTimeShift': 0.0, 'MeaTimeDelay': 0.0}},
    )
    assert getGateTimeTable(gex, 'Channel1').shape == (37, 3)
    # each channel slices the shared table down to its own gate count
    assert getGateTimesFromGEX(gex, 'Channel1').shape == (28, 3)
    assert getGateTimesFromGEX(gex, 'Channel2').shape == (37, 3)


# --- per-moment convention (SkyTEM 306HP, 312HP) -------------------------------

def test_per_moment_tables_selected_by_transmitter_moment():
    gex = _gex(
        {'GateTimeLM': _table(25), 'GateTimeHM': _table(41, start=5e-6)},
        {'Channel1': {'NoGates': 25, 'TransmitterMoment': 'LM',
                      'GateTimeShift': 0.0, 'MeaTimeDelay': 0.0},
         'Channel2': {'NoGates': 41, 'TransmitterMoment': 'HM',
                      'GateTimeShift': 0.0, 'MeaTimeDelay': 0.0}},
    )
    assert getGateTimesFromGEX(gex, 'Channel1').shape == (25, 3)
    assert getGateTimesFromGEX(gex, 'Channel2').shape == (41, 3)
    # the two moments must resolve to genuinely different tables
    assert not np.allclose(getGateTimeTable(gex, 'Channel1')[0, 0],
                           getGateTimeTable(gex, 'Channel2')[0, 0])


def test_shift_and_delay_still_applied_for_per_moment():
    gex = _gex(
        {'GateTimeLM': _table(25)},
        {'Channel1': {'NoGates': 25, 'TransmitterMoment': 'LM',
                      'GateTimeShift': 1e-7, 'MeaTimeDelay': 2e-7}},
    )
    got = getGateTimesFromGEX(gex, 'Channel1')
    assert np.allclose(got, _table(25) + 1e-7 + 2e-7)


# --- failure reporting ---------------------------------------------------------

def test_missing_table_names_what_was_looked_for():
    gex = _gex(
        {'GateTimeHM': _table(41)},
        {'Channel1': {'NoGates': 25, 'TransmitterMoment': 'LM',
                      'GateTimeShift': 0.0, 'MeaTimeDelay': 0.0}},
    )
    with pytest.raises(KeyError) as exc:
        getGateTimesFromGEX(gex, 'Channel1')
    msg = str(exc.value)
    assert 'GateTimeLM' in msg      # what it wanted
    assert 'GateTimeHM' in msg      # what was actually there


def test_missing_transmitter_moment_is_reported():
    gex = _gex(
        {'GateTimeLM': _table(25)},
        {'Channel1': {'NoGates': 25, 'GateTimeShift': 0.0, 'MeaTimeDelay': 0.0}},
    )
    with pytest.raises(KeyError) as exc:
        getGateTimesFromGEX(gex, 'Channel1')
    assert 'TransmitterMoment' in str(exc.value)
