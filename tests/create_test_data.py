"""Extract 200-sounding subsets from real data for test fixtures.
Run once: .venv/bin/python tests/create_test_data.py
"""
import libaarhusxyz
import numpy as np
import re
from emeraldprocessing.tem.dataIO import getGateTimesFromGEX
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)


def parse_gate_times_from_gex(gex_path):
    """Parse gate center times from a GEX file (works for HeliTEM GEX
    that libaarhusxyz.GEX can't handle due to missing TransmitterMoment)."""
    gate_times = []
    with open(gex_path) as f:
        for line in f:
            m = re.match(r'GateTime\d+=\s+([\d.eE+-]+)', line.strip())
            if m:
                gate_times.append(float(m.group(1)))
    return np.array(gate_times)


# HeliTEM
print("Loading HeliTEM data...")
xyz_h = libaarhusxyz.XYZ('/tmp/helitem_converted/2501013_prelim_beryl.xyz')
helitem_key = [k for k in xyz_h.layer_data if 'gate' in k.lower()][0]
helitem_layer = xyz_h.layer_data[helitem_key]
print(f"  Key: {helitem_key}, shape: {helitem_layer.shape}")

# Gate center times parsed from the GEX produced by emerald-helitem-converter
helitem_gt = parse_gate_times_from_gex('/tmp/helitem_converted/2501013_prelim_beryl.gex')
assert len(helitem_gt) == helitem_layer.shape[1], (
    f"GEX has {len(helitem_gt)} gates but data has {helitem_layer.shape[1]} columns")

dlog = np.diff(np.log10(helitem_gt))
print(f"  Gate times: {helitem_gt[0]*1e3:.3f}ms – {helitem_gt[-1]*1e3:.3f}ms")
print(f"  Δlog10(t) range: {dlog.min():.4f} – {dlog.max():.4f}")
print(f"  Total decades: {np.log10(helitem_gt[-1]) - np.log10(helitem_gt[0]):.4f}")

np.savez(DATA_DIR / 'helitem_200.npz',
         layer_data=helitem_layer.iloc[:200].values,
         gate_times=helitem_gt, scalefactor=1.0)
print(f"  Saved helitem_200.npz")

# SkyTEM LM
print("\nLoading SkyTEM data...")
xyz_s = libaarhusxyz.XYZ('/tmp/skytem_test_data/17_Test-5-lines-EJH-20260204-env-measured.xyz')
gex_s = libaarhusxyz.GEX('/tmp/skytem_test_data/17_Test-5-lines-EJH-20260204-env-measured.gex')
skytem_key = [k for k in xyz_s.layer_data if 'gate' in k.lower()][0]
print(f"  Key: {skytem_key}, shape: {xyz_s.layer_data[skytem_key].shape}")

skytem_gt = getGateTimesFromGEX(gex_s, 'Channel1')[:, 0]
dlog = np.diff(np.log10(skytem_gt[skytem_gt > 0]))
print(f"  Gate times: {skytem_gt[0]*1e6:.1f}μs – {skytem_gt[-1]*1e3:.3f}ms")
print(f"  Δlog10(t) range: {dlog.min():.4f} – {dlog.max():.4f}")

np.savez(DATA_DIR / 'skytem_lm_200.npz',
         layer_data=xyz_s.layer_data[skytem_key].iloc[:200].values,
         gate_times=skytem_gt, scalefactor=1e-12)
print(f"  Saved skytem_lm_200.npz")

print("\nDone!")
