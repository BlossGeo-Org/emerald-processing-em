import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


def _make_mock_processing(npz_path, data_key='Gate_Ch01'):
    """Build a mock processing object from saved real data."""
    d = np.load(npz_path)
    mock = SimpleNamespace()
    mock.xyz = SimpleNamespace()
    mock.xyz.layer_data = {data_key: pd.DataFrame(d['layer_data'])}
    mock.xyz.model_info = {'scalefactor': float(d['scalefactor'])}
    mock.GateTimes = {data_key: d['gate_times']}
    return mock


@pytest.fixture(scope='session')
def helitem_processing():
    return _make_mock_processing(DATA_DIR / 'helitem_200.npz')


@pytest.fixture(scope='session')
def skytem_processing():
    return _make_mock_processing(DATA_DIR / 'skytem_lm_200.npz')
