# Plan: Add Decimation Option to `moving_average_filter`

## Motivation

The upstream SkyTEM / NRD survey processing in Aarhus Workbench exports AEM data at
a much coarser spacing (~25 m) than the raw acquisition rate (~2.6 m), having applied
a moving-average / stacking filter and then sub-sampled the result.  The current
`moving_average_filter` step in nagelfluh smooths data in place but leaves the sounding
count unchanged.  To match the upstream workflow (and to reduce inversion cost) we need
a **decimation option** that retains every N-th sounding after averaging.

Observed values for the NRD 2018 survey (line 409001):

| Property | Raw | Upstream processed |
|----------|-----|--------------------|
| Soundings (line 409001 segment) | 5 003 | ~350 |
| Median spacing | 2.6 m | 25.6 m |
| Decimation factor | — | ~10× |

---

## Proposed API change

Add two **optional** parameters to `moving_average_filter`
(`emeraldprocessing/tem/corrections.py`):

```python
def moving_average_filter(
    processing: pipeline.ProcessingData,
    filter_dict: MovingAverageFilterDict = ...,
    averaging_method: str = 'hybrid',
    min_valid_fraction: float = 0.35,
    decimation_factor: int = 1,        # NEW — keep every N-th sounding (1 = no decimation)
    target_spacing_m: float = None,    # NEW — derive N from target spacing; overrides decimation_factor
    verbose: bool = False,
):
```

When `decimation_factor=1` and `target_spacing_m=None` (the defaults), behaviour is
identical to the current implementation.  No existing call sites break.

### Parameter semantics

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `decimation_factor` | `int` | `1` | Keep every N-th sounding after averaging.  `1` = keep all (no decimation). |
| `target_spacing_m` | `float \| None` | `None` | If set, compute `N = round(target_spacing_m / median_spacing)` per line and use that as the stride.  Takes precedence over `decimation_factor`. |

`target_spacing_m` is the more user-facing parameter; `decimation_factor` is the
low-level escape hatch.

---

## Implementation plan

### 1. Compute per-line stride

In `moving_average_filter`, after `lines = utils.splitData_lines(data, ...)`, resolve
the effective stride for each line:

```python
def _resolve_stride(line_data, decimation_factor, target_spacing_m):
    if target_spacing_m is not None:
        spacing = utils.estimateInlineSamplig(line_data)
        if spacing and spacing > 0:
            return max(1, round(target_spacing_m / spacing))
    return max(1, decimation_factor)
```

`utils.estimateInlineSamplig` already exists and returns the median along-line
spacing in metres.

### 2. Apply decimation per line after averaging

Inside `movingAverageFilterLine` (or immediately after the call to it), subsample
the line using `utils.filtXYZ`:

```python
stride = _resolve_stride(line_data, decimation_factor, target_spacing_m)
if stride > 1:
    n = len(line_data.flightlines)
    keep = np.zeros(n, dtype=bool)
    keep[::stride] = True
    line_data = utils.filtXYZ(line_data, keep)
```

`filtXYZ` deep-copies the object and calls `drop_filt_XYZ(~filt)` on both
`flightlines` and every `layer_data` DataFrame, then resets integer indices.
It already handles all required bookkeeping.

### 3. Signature of `movingAverageFilterLine`

Pass the stride through:

```python
def movingAverageFilterLine(lineData, filter_dict, averaging_method='hybrid',
                            min_valid_fraction=0.35, stride=1, verbose=False):
    ...
    # existing averaging logic unchanged
    ...
    if stride > 1:
        n = len(lineData.flightlines)
        keep = np.zeros(n, dtype=bool)
        keep[::stride] = True
        utils.drop_filt_XYZ(lineData, ~keep)   # mutate in-place (matches current pattern)
```

`drop_filt_XYZ` mutates in-place (no copy), consistent with how `movingAverageFilterLine`
currently modifies `lineData` in place before `merge_lines` rebuilds the full dataset.

### 4. Docstring additions

Extend the `moving_average_filter` docstring:

```
decimation_factor : int, optional
    Sub-sample the averaged data by keeping every N-th sounding (default 1 = no
    decimation).  Applied per line after the moving-average step so the filter
    window always operates on the full-resolution data.

target_spacing_m : float, optional
    Target along-line sounding spacing in metres.  If provided, the decimation
    stride is computed automatically as ``round(target_spacing_m / median_spacing)``
    per flight line.  Takes precedence over ``decimation_factor``.
    Use ``utils.estimateInlineSamplig`` to inspect the current spacing before
    choosing a target.
```

### 5. JSON schema (`pipeline_step` entry)

The Nagelfluh JSON schema for `Moving average filter` is auto-generated from the
function signature via `scripts/make_json_schema.py`.  Re-running that script after
the code change will emit:

```json
"decimation_factor": {
  "default": 1,
  "description": "Sub-sample the averaged data by keeping every N-th sounding …",
  "type": "integer",
  "x-python-type": "builtins.int"
},
"target_spacing_m": {
  "description": "Target along-line sounding spacing in metres …",
  "type": "number",
  "x-python-type": "builtins.float"
}
```

No new `parameter_types` entry is needed — both types are standard Python scalars.

---

## Files to change

| File | Change |
|------|--------|
| `emeraldprocessing/tem/corrections.py` | Add `decimation_factor`, `target_spacing_m` params to `moving_average_filter`; add `_resolve_stride` helper; pass stride to `movingAverageFilterLine` |
| `emeraldprocessing/tem/corrections.py` | Add `stride` param and post-averaging decimation block to `movingAverageFilterLine` |
| `tests/test_averaging.py` | Add tests: stride=1 (identity), stride=N (sounding count), target_spacing_m (auto-stride), edge cases (line shorter than stride) |

No changes to `parameter_types.py`, `utils.py`, or any schema files (schema is
auto-generated).

---

## Edge cases to handle

- **Line shorter than stride**: `keep[::stride]` always keeps at least the first
  sounding, so even a 1-sounding line is not dropped entirely.
- **stride > line length**: `keep[::stride]` returns a single `True` at index 0.
  The line survives with one sounding.
- **Non-integer ratio**: `round()` gives the closest integer stride.  Log the
  actual computed spacing and stride if `verbose=True`.
- **`target_spacing_m` smaller than actual spacing**: resolves to stride=1
  (no-op), consistent with the `max(1, ...)` guard.
- **Multiple lines with different natural spacings**: stride is computed
  independently per line, so each line decimates to approximately the target
  spacing regardless of along-line speed variation.

---

## Testing strategy

```python
# tests/test_averaging.py additions

def test_no_decimation(sample_processing):
    n_before = len(sample_processing.xyz.flightlines)
    moving_average_filter(sample_processing, decimation_factor=1)
    assert len(sample_processing.xyz.flightlines) == n_before

def test_decimation_factor(sample_processing):
    n_before = len(sample_processing.xyz.flightlines)
    moving_average_filter(sample_processing, decimation_factor=3)
    n_after = len(sample_processing.xyz.flightlines)
    assert n_after == pytest.approx(n_before / 3, abs=2)

def test_target_spacing(sample_processing):
    # sample data has 2.5 m spacing → target 25 m should give ~10× decimation
    moving_average_filter(sample_processing, target_spacing_m=25.0)
    n_after = len(sample_processing.xyz.flightlines)
    expected = len(sample_processing.xyz.flightlines) // 10
    # allow ±1 per line due to rounding
    assert abs(n_after - expected) <= n_lines

def test_decimation_preserves_layer_data_shape(sample_processing):
    n_gates_before = {k: v.shape[1] for k, v in sample_processing.xyz.layer_data.items()}
    moving_average_filter(sample_processing, decimation_factor=5)
    for k, ncols in n_gates_before.items():
        assert sample_processing.xyz.layer_data[k].shape[1] == ncols

def test_decimation_resets_index(sample_processing):
    moving_average_filter(sample_processing, decimation_factor=4)
    fl = sample_processing.xyz.flightlines
    assert list(fl.index) == list(range(len(fl)))
    for k, df in sample_processing.xyz.layer_data.items():
        assert list(df.index) == list(range(len(df)))
```
