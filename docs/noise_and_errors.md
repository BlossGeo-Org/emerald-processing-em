# Noise and Error Modelling for AEM Synthetic Data

This document describes the noise/error pipeline used when generating synthetic forward-modelled
data for SkyTEM 304 (and similar dual-moment helicopter TEM) systems, based on code inspection,
empirical calibration against real survey data, and published AEM literature.

---

## Overview of filters

Two filters work together to add realistic noise to synthetic data:

1. **`STD error: Replace from GEX`** (`corrections.py:add_replace_gex_std_error`) — sets the
   per-gate STD (relative fraction) stored in `STD_Ch##`.
2. **`Add noise realization`** (`corrections.py:add_noise_realization`) — draws a Gaussian noise
   sample scaled by the STD and adds it to the signal in `Gate_Ch##`.

The STD throughout the pipeline is a **dimensionless relative fraction** (e.g. 0.03 = 3%). It is
not an absolute value.

---

## `STD error: Replace from GEX` — operating modes

The filter has four modes depending on which optional parameters are supplied:

| Parameters | Formula | Notes |
|---|---|---|
| neither (default) | $\sigma = \sigma_\text{GEX}$ | Flat fraction from `Channel{n}/UniformDataSTD`; no signal dependence |
| `relative_noise_fraction` only | $\sigma = \sqrt{(\sigma_\text{GEX}/\|d\|)^2 + r^2}$ | Quadrature: absolute floor (GEX value) + relative component |
| `noise_level_1ms` only | $\sigma = N(t)/\|d\|$ | Time-varying absolute floor, signal-normalised |
| both | $\sigma = \sqrt{(N(t)/\|d\|)^2 + r^2}$ | Full model: time-varying floor + relative |

where:
- $\sigma_\text{GEX}$ = `Channel{n}/UniformDataSTD` from the GEX file (an absolute value in data
  units — see section below)
- $r$ = `relative_noise_fraction` (e.g. 0.03)
- $N(t) = N_0 \cdot (t / 1\,\text{ms})^\alpha$ — time-dependent absolute noise floor
  (`noise_level_1ms` = $N_0$, `noise_exponent` = $\alpha$)
- $\|d\|$ = absolute signal amplitude; clipped at the noise floor to avoid division by zero

---

## `UniformDataSTD` in the GEX file

### What it is

`Channel{n}/UniformDataSTD` is a single scalar per channel (e.g. `0.03`) set by the survey
operator in the GEX file. It appears in all GEX files in this repository with a value of 0.03.

### What it is NOT

- There is no per-gate noise floor in the GEX format. The format only stores this one scalar
  per channel.
- The GEX general section has a `CalculateRawDataSTD` flag (0 or 1) which is an Aarhus Workbench
  processing instruction; it is not a noise value and is not read by this codebase.

### Physical meaning — ambiguity

No public documentation (Aarhus Workbench manual, SkyTEM specifications, or online sources)
defines what `UniformDataSTD` represents physically. Two interpretations exist:

**Interpretation A — relative fraction (probable original intent):**
The value 0.03 means 3% of the signal. This is how Aarhus Workbench and the `forward_process.py`
file use it (`gate_df * 0 + uniform_std`), and it matches the conventional community default of
3% relative uncertainty for AEM inversion.

**Interpretation B — absolute noise floor (used in `corrections.py`):**
The `add_replace_gex_std_error` function treats `UniformDataSTD` as an absolute value in data
units (V/(A·m⁴)), divides it by the signal, and clips the signal at the GEX value. This is
consistent with using it as a noise floor amplitude rather than a fraction.

**Consequence:** when using `relative_noise_fraction` mode (without `noise_level_1ms`), the
GEX `UniformDataSTD` is used as an absolute floor in the quadrature formula. If the GEX stores
0.03 as a relative fraction, this misinterpretation makes the floor effectively signal-dependent
in an unintended way. The `noise_level_1ms` mode avoids this ambiguity entirely.

---

## Absolute noise floor: units and parameter value for SkyTEM 304

### Units of `noise_level_1ms`

**`noise_level_1ms` is in V/m²** (dB/dt normalised by receiver effective area, but *not* by
transmitter dipole moment).

Derivation: `make_noise_df` (`utils.py:1197`) computes:

```python
noise = (gate_time / 1e-3) ** noise_exponent * noise_level_1ms
noise_df = noise_df / data.model_info['scalefactor']          # scalefactor = 1.0 in all datasets
noise_df = noise_df / processing.ApproxDipoleMoment[data_key] # A·m²  →  output in V/(A·m⁴)
```

With `scalefactor = 1.0` (confirmed in all XYZ files in this repository), the only normalisation
applied to `noise_level_1ms` is the transmitter dipole moment. For the output to be in V/(A·m⁴),
`noise_level_1ms` must be in V/m².

Note: the docstring in `corrections.py:460` incorrectly states "V/(A·m⁴)" for this parameter;
that describes the *output* of `make_noise_df`, not the input.

### SkyTEM 304 system parameters

From the SkyTEM 304 specification sheet and the GEX file
(`20201231_20023_IVF_SkyTEM304_SKB.gex`):

| Parameter | LM (Ch1) | HM (Ch2) |
|---|---|---|
| Transmitter turns | 1 | 4 |
| Transmitter loop area | 342 m² | 342 m² |
| Approximate current | 8.8 A | 113.7 A |
| Dipole moment (`ApproxDipoleMoment`) | 3,010 A·m² | 155,578 A·m² |
| Z receiver effective area | 105 m² | 105 m² |
| LM gate range | 5 µs – 876 µs | — |
| HM gate range | — | 70 µs – 8,900 µs |

### Unit conversion: V/m² ↔ V/(A·m⁴)

Published AEM noise floors are often quoted in fully-normalised units V/(A·m⁴) (divided by both
receiver area and transmitter moment). The `noise_level_1ms` parameter uses the intermediate unit
V/m² (divided by receiver area only). The conversion is:

$$N_0\,[\text{V/(A·m}^4)] = \frac{N_0\,[\text{V/m}^2]}{M_\text{tx}\,[\text{A·m}^2]}$$

For SkyTEM 304:

| Channel | M_tx (A·m²) | noise_level_1ms (V/m²) → V/(A·m⁴) |
|---|---|---|
| LM (Ch1) | 3,010 | divide by 3,010 |
| HM (Ch2) | 155,578 | divide by 155,578 |

### Empirical calibration of `noise_level_1ms`

Using the California Central Valley SkyTEM 304 dataset
(`aem_processed_data_foothill_central_valley.measured.xyz`), the absolute noise floor was
back-calculated from late gates where the signal has decayed enough for the floor to dominate
over the 3% relative term:

$$N_\text{abs} = \sqrt{\sigma_\text{obs}^2 - 0.03^2} \times |d| \times M_\text{tx}$$

#### LM (Ch1) — gates 21–28, t ≈ 550–876 µs

These gates are already close to the 1 ms reference, so the t^{-0.5} correction is less than 10%.

| Gate | STD_obs | Signal V/(A·m⁴) | N_abs V/m² | N_abs V/(A·m⁴) |
|------|---------|-----------------|------------|----------------|
| 24 | 5.1% | 4.1e-12 | 5.1e-10 | 1.7e-13 |
| 25 | 6.2% | 3.0e-12 | 4.9e-10 | 1.6e-13 |
| 26 | 9.9% | 2.0e-12 | 5.8e-10 | 1.9e-13 |
| 27 | 13.0% | 1.6e-12 | 6.1e-10 | 2.0e-13 |
| 28 | 14.3% | 1.1e-12 | 4.6e-10 | 1.5e-13 |

Converges to **~5×10⁻¹⁰ V/m² (~1.7×10⁻¹³ V/(A·m⁴))** at t ≈ 1 ms.

#### HM (Ch2) — gates 31–35, t ≈ 3–8 ms

The floor emerges later because HM signals are larger and decay more slowly.  Because these
gates are well past 1 ms, the t^{-0.5} correction to the reference point is significant.

| Gate | STD_obs | Signal V/(A·m⁴) | N_abs V/m² | N_abs V/(A·m⁴) |
|------|---------|-----------------|------------|----------------|
| 31 | 3.44% | 2.9e-13 | 7.6e-10 | 4.9e-15 |
| 32 | 3.93% | 1.8e-13 | 7.3e-10 | 4.7e-15 |
| 33 | 4.88% | 1.1e-13 | 6.9e-10 | 4.4e-15 |
| 34 | 7.01% | 7.4e-14 | 7.3e-10 | 4.7e-15 |
| 35 | 10.72% | 4.8e-14 | 7.6e-10 | 4.9e-15 |

At these late gates (estimated t ≈ 3–8 ms), $N_\text{abs}$ converges to **~7.5×10⁻¹⁰ V/m²
(~4.7×10⁻¹⁵ V/(A·m⁴))**. Extrapolating back to t=1 ms via $N_0 = N(t)\cdot(t/1\,\text{ms})^{+0.5}$:

$$N_0^\text{HM} \approx 7.5\times10^{-10} \times \sqrt{t_\text{ms}} \approx 1.5\times10^{-9}\,\text{V/m}^2$$

(using a representative t ≈ 4 ms; the result is ~1.0–2.0×10⁻⁹ V/m² across the window).

The receiver is the same 105 m² Z coil for both moments.  The HM reference value being ~3× higher
than LM is plausible: HM gates extend to much later times where slow environmental noise
(atmospheric, cultural) accumulates beyond the receiver thermal floor.

This is consistent with published values for SkyTEM and similar helicopter TEM systems of
0.2–5 nV/m² in quiet environments (Auken et al. 2009; Christiansen & Auken 2012).

### Cross-check against circulated "industry" values

A set of values sometimes cited as "standard" SkyTEM 304 reference noise floors (with no
traceable published source) is:

- LM: ~2.5×10⁻¹² V/(A·m⁴) at 1 ms
- HM: ~2.5×10⁻¹³ V/(A·m⁴) at 1 ms

**These values are inconsistent with the empirical data and should not be used.**

At LM gate 28 (signal = 1.09×10⁻¹² V/(A·m⁴)), a floor of 2.5×10⁻¹² V/(A·m⁴) implies 229%
relative noise — the data would be entirely noise and should have been culled.  The observed STD
is 14%, matching the empirical floor of ~1.7×10⁻¹³ V/(A·m⁴):

| Floor source | LM V/(A·m⁴) | Implied STD at gate 28 | Consistent with data? |
|---|---|---|---|
| Empirical (this work) | 1.7×10⁻¹³ | ~15% | ✓ matches observed ~14% |
| Circulated "standard" | 2.5×10⁻¹² | ~229% | ✗ data would be unusable |

### Recommended parameter values

#### LM (Ch1)

```json
{
  "STD error: Replace from GEX": {
    "channel": 1,
    "noise_level_1ms": 5e-10,
    "noise_exponent": -0.5,
    "relative_noise_fraction": 0.03
  }
}
```

| Parameter | Value | V/(A·m⁴) equivalent | Rationale |
|---|---|---|---|
| `noise_level_1ms` | `5e-10` V/m² | ~1.7×10⁻¹³ | Empirically derived; quiet/rural SkyTEM 304 LM survey |
| `noise_exponent` | `-0.5` | — | Gate averaging with logarithmic spacing: noise ∝ t^{-1/2} |
| `relative_noise_fraction` | `0.03` | — | 3% relative uncertainty (positioning, altitude, tx moment) |

For noisier environments scale up to `1e-9` V/m² (~3×10⁻¹³ V/(A·m⁴)).

#### HM (Ch2)

```json
{
  "STD error: Replace from GEX": {
    "channel": 2,
    "noise_level_1ms": 1.5e-9,
    "noise_exponent": -0.5,
    "relative_noise_fraction": 0.03
  }
}
```

| Parameter | Value | V/(A·m⁴) equivalent | Rationale |
|---|---|---|---|
| `noise_level_1ms` | `1.5e-9` V/m² | ~9.6×10⁻¹⁵ | Empirically derived from HM late gates, extrapolated to t=1ms reference |
| `noise_exponent` | `-0.5` | — | Same gate-averaging argument as LM |
| `relative_noise_fraction` | `0.03` | — | 3% relative uncertainty (same as LM) |

For noisier environments scale up to `3e-9` V/m² (~1.9×10⁻¹⁴ V/(A·m⁴)).

Note: the HM `noise_level_1ms` is higher than LM because the reference value is dominated by
environmental noise at the very late gates (>3 ms) rather than purely receiver thermal noise.
If using HM data only out to ~2 ms, a lower value of `5e-10` V/m² (same as LM) may be more
appropriate.

---

## Physical basis of the noise model

### Two-component noise decomposition

AEM data error is standardly decomposed into two independent components (Auken et al. 2009;
Viezzoli et al. 2008):

**Additive (absolute) noise floor** — thermal (Johnson) noise from receiver coil and electronics.
Approximately constant in V/m² (i.e. gate-independent for the raw electronics noise). Dominates
at late gates where the decaying signal is small.

**Multiplicative (relative) noise** — positioning error (GPS altitude uncertainty ~1–2 m),
transmitter moment variation, bird attitude. Proportional to signal amplitude. Dominates at early
to mid gates. Typical value: 3–5% for SkyTEM.

These combine in quadrature:
$$\sigma = \sqrt{\left(\frac{N(t)}{|d|}\right)^2 + r^2}$$

This is the standard data error model used for diagonal-covariance 1D inversion in AarhusInv/SCI.

### The `noise_exponent = -0.5` and gate averaging

Raw electronics noise is approximately white (flat in V/m²). After gate averaging, the noise in
each gate scales as:

$$\sigma_\text{gate}(t) \propto \frac{1}{\sqrt{\Delta t(t)}}$$

For SkyTEM's logarithmically-spaced gates, $\Delta t \propto t$, giving $\sigma_\text{gate}(t)
\propto t^{-1/2}$. This is the physical justification for `noise_exponent = -0.5`. The reference
point at 1 ms is arbitrary (any reference time works, as long as `noise_level_1ms` is consistent).

Verification from the SkyTEM 304 spec sheet gate table (LM):
- Gate at t=5 µs: width 1.6 µs → 1/√1.6 ≈ 0.79
- Gate at t=876 µs: width 202 µs → 1/√202 ≈ 0.070
- Ratio: 0.79/0.070 ≈ 11.3; predicted by t^{-1/2}: (5/876)^{-1/2} ≈ 13.2  ✓ (order of magnitude)

### What the model omits

- **Gate-to-gate correlation**: real AEM gates have overlapping filter responses; the model
  assumes independent noise. Acceptable for 1D inversion.
- **Non-Gaussian tails**: lightning strikes, cultural interference, helicopter vibration.
- **Altitude dependence**: signal (and thus noise in absolute terms) scales with altitude;
  not modelled here.
- **Negative data / sign reversals**: the `.clip(lower=noise_floor)` in the code masks near-zero
  or negative signals. Late gates in polarisable ground or certain geometries can legitimately go
  negative.

---

## Noise realization (`Add noise realization`)

After setting the STD, the `Add noise realization` filter (`corrections.py:517`) adds a Gaussian
sample:

```python
noise = rng.normal(0.0, std_df.values * np.abs(signal.values))
data.layer_data[data_key] = signal + noise
```

The STD (relative fraction) is converted back to absolute noise by multiplying by `|signal|`
before sampling. A `seed` parameter enables reproducible synthetic datasets.

**Workflow for synthetic data:**

```json
[
  {"Forward model": { ... }},
  {"STD error: Replace from GEX": {"channel": 1, "noise_level_1ms": 5e-10, "noise_exponent": -0.5, "relative_noise_fraction": 0.03}},
  {"Add noise realization": {"channel": 1}},
  {"Inversion": { ... }}
]
```

---

## References

- Auken, E. et al. (2009). *Layered and laterally constrained 2D inversion of resistivity data.*
  Geophysical Prospecting.
- Christiansen, A.V. & Auken, E. (2012). *A global measure for depth of investigation.*
  Geophysics.
- Sørensen, K.I. & Auken, E. (2004). *SkyTEM — a new high-resolution helicopter transient
  electromagnetic system.* Exploration Geophysics, 35(3).
- Viezzoli, A. et al. (2008). *Quasi-3D modeling of airborne TEM data by spatially constrained
  inversion.* Geophysics.
- SkyTEM Surveys APS. *Specifications of the SkyTEM304 System (50 Hz)*, 2014.
