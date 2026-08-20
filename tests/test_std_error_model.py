"""
Tests for the GEX / noise-model STD error steps.

Locks in the semantics of add_replace_gex_std_error and
add_std_error_from_noise_model against three filed issues:

  #63 the combination rule is quadrature, and the docstring must say so
  #70 UniformDataSTD is a dimensionless fraction, never an absolute amplitude
  #38 relative_noise_fraction is a quadrature floor, not a scale factor

#63 and #70 describe a single combined function that has since been split in
two; these tests assert the properties the issues were really about, so the
old behavior cannot come back under the new structure.
"""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from emeraldprocessing.tem import corrections

GEX_STD = 0.03
N_SOUNDINGS = 40
N_GATES = 12


def _make_processing(scalefactor=1.0):
    """Mock with dB/dt spanning the realistic 1e-9 .. 1e-15 V/(A*m^4) range.

    Stored values are divided by scalefactor, mirroring how Workbench exports
    scaled integers; the physical magnitude is identical either way.
    """
    gate_times = np.logspace(-5, -3, N_GATES)
    decay = np.logspace(-9, -15, N_GATES)
    physical = np.tile(decay, (N_SOUNDINGS, 1))
    physical *= np.linspace(0.8, 1.2, N_SOUNDINGS)[:, None]

    p = SimpleNamespace()
    p.xyz = SimpleNamespace()
    p.xyz.layer_data = {'Gate_Ch01': pd.DataFrame(physical / scalefactor)}
    p.xyz.flightlines = pd.DataFrame(index=range(N_SOUNDINGS))
    p.xyz.model_info = {'scalefactor': scalefactor}
    p.gex = SimpleNamespace(gex_dict={'Channel1': {'UniformDataSTD': GEX_STD}})
    p.GateTimes = {'Gate_Ch01': gate_times}
    p.ApproxDipoleMoment = {'Gate_Ch01': 1.0}
    return p


def _std(processing):
    return processing.xyz.layer_data['STD_Ch01'].values


# --------------------------------------------------------------------------
# #70 : UniformDataSTD is a fraction, not an amplitude
# --------------------------------------------------------------------------

def test_gex_std_is_uniform_fraction():
    p = _make_processing()
    corrections.add_replace_gex_std_error(p, channel=1)
    np.testing.assert_allclose(_std(p), GEX_STD)


@pytest.mark.parametrize('relative_noise_fraction', [0.01, 0.03, 0.10])
def test_relative_fraction_alone_does_not_give_unit_uncertainty(relative_noise_fraction):
    """Regression for #70: STD must not collapse to ~1.0 (100% uncertainty).

    The old code divided UniformDataSTD by the signal clipped at
    UniformDataSTD. Since dB/dt is far below 0.03 in physical units, the
    denominator became 0.03 for every datum and the ratio was exactly 1.0.
    """
    p = _make_processing()
    corrections.add_replace_gex_std_error(p, channel=1)
    corrections.add_std_error_from_noise_model(
        p, channel=1, relative_noise_fraction=relative_noise_fraction)

    expected = np.sqrt(GEX_STD ** 2 + relative_noise_fraction ** 2)
    np.testing.assert_allclose(_std(p), expected)
    assert np.all(_std(p) < 0.5), "STD collapsed toward 100% uncertainty"


def test_std_is_independent_of_data_scalefactor():
    """The root of #70 was a fraction compared against data amplitudes.

    Any such comparison makes the answer depend on how the data happen to be
    scaled. The same physical data must give the same STD either way.
    """
    scaled, physical = _make_processing(scalefactor=1e-12), _make_processing(scalefactor=1.0)
    for p in (scaled, physical):
        corrections.add_replace_gex_std_error(p, channel=1)
        corrections.add_std_error_from_noise_model(
            p, channel=1, noise_level_1ms=5e-10, relative_noise_fraction=0.02)

    np.testing.assert_allclose(_std(scaled), _std(physical), rtol=1e-12)


# --------------------------------------------------------------------------
# #63 : quadrature, not geometric mean
# --------------------------------------------------------------------------

def test_combination_is_quadrature_not_geometric_mean():
    """Combining two noise sources must exceed both, not land between them."""
    p = _make_processing()
    corrections.add_replace_gex_std_error(p, channel=1)
    corrections.add_std_error_from_noise_model(
        p, channel=1, relative_noise_fraction=0.04)

    combined = _std(p)
    np.testing.assert_allclose(combined, np.sqrt(GEX_STD ** 2 + 0.04 ** 2))

    assert np.all(combined > GEX_STD)
    assert np.all(combined > 0.04)
    # The geometric mean would be sqrt(0.03 * 0.04) = 0.0346, below the larger
    # component — the physically wrong answer the old docstring described.
    assert np.all(combined > np.sqrt(GEX_STD * 0.04))


def test_docstrings_do_not_claim_a_geometric_mean():
    """The docstring reaches the product UI via swaggerspect (#63)."""
    for fn in (corrections.add_replace_gex_std_error,
               corrections.add_std_error_from_noise_model):
        doc = (fn.__doc__ or '').lower()
        assert 'geometric mean' not in doc
        assert 'geomean' not in doc


# --------------------------------------------------------------------------
# #38 : relative_noise_fraction is a floor, not a scale
# --------------------------------------------------------------------------

def test_relative_fraction_is_monotonic_non_decreasing():
    """Lowering relative_noise_fraction can never raise the STD.

    #38 reported the median rising ~4x on lowering it from 0.03 to 0.01.
    That cannot originate here: every term enters as +r^2 under a square root.
    """
    results = []
    for r in (0.0001, 0.01, 0.03, 0.10):
        p = _make_processing()
        corrections.add_replace_gex_std_error(p, channel=1)
        corrections.add_std_error_from_noise_model(
            p, channel=1, noise_level_1ms=5e-10, relative_noise_fraction=r)
        results.append(_std(p))

    for lower, higher in zip(results, results[1:]):
        assert np.all(lower <= higher + 1e-12)


def test_relative_fraction_acts_as_a_floor():
    """The result never falls below relative_noise_fraction or the base STD."""
    p = _make_processing()
    corrections.add_replace_gex_std_error(p, channel=1)
    corrections.add_std_error_from_noise_model(
        p, channel=1, relative_noise_fraction=0.25)

    assert np.all(_std(p) >= 0.25)
    assert np.all(_std(p) >= GEX_STD)


def test_relative_fraction_is_nearly_invisible_when_dominated():
    """Quadrature hides any term much smaller than the others (#38).

    This is the behavior that reads as 'it does not scale the error model':
    tripling the parameter moves the result by a fraction of a percent.
    """
    stds = {}
    for r in (0.01, 0.03):
        p = _make_processing()
        corrections.add_replace_gex_std_error(p, channel=1)
        corrections.add_std_error_from_noise_model(
            p, channel=1, noise_level_1ms=5e-10, relative_noise_fraction=r)
        stds[r] = np.median(_std(p))

    assert stds[0.03] > stds[0.01]
    relative_change = (stds[0.03] - stds[0.01]) / stds[0.01]
    assert relative_change < 0.01, (
        f"expected the dominated term to be near-invisible, "
        f"got a {relative_change:.1%} change")


def test_noise_model_populates_std_when_missing():
    """The step is usable on its own; it seeds the base STD from the GEX."""
    p = _make_processing()
    assert 'STD_Ch01' not in p.xyz.layer_data

    corrections.add_std_error_from_noise_model(
        p, channel=1, relative_noise_fraction=0.04)

    np.testing.assert_allclose(_std(p), np.sqrt(GEX_STD ** 2 + 0.04 ** 2))
