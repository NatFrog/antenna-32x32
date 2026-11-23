"""Convert drive-table / result CSVs into a single compressed dataset file.

Output: data/dataset.npz with
    weights   complex64 [N, 32, 32]   element excitation (row_i, col_j)
    gain_db   float32   [N, 181, 360] dB(GainTotal) on (theta, phi) grid
    theta_deg float32   [181]
    phi_deg   float32   [360]
    n_beams   int64     [N]
    names     str       [N]
"""

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DRIVE_DIR = ROOT / "drive_tables"
RESULT_DIR = ROOT / "Result_drive_tables_32x32"
OUT = ROOT / "data" / "dataset.npz"
N = 32


def load_pair(name: str):
    d = pd.read_csv(DRIVE_DIR / f"{name}.csv")
    w = np.zeros((N, N), np.complex64)
    w[d.row_i.values, d.col_j.values] = d.weight_re.values + 1j * d.weight_im.values

    r = pd.read_csv(RESULT_DIR / f"Result_{name}.csv")
    r.columns = ["theta", "phi", "g"]
    theta = np.sort(r.theta.unique())
    phi = np.sort(r.phi.unique())
    ti = np.searchsorted(theta, r.theta.values)
    pi = np.searchsorted(phi, r.phi.values)
    g = np.full((len(theta), len(phi)), np.nan, np.float32)
    g[ti, pi] = r.g.values
    return name, w, g, theta, phi


def main():
    names = sorted(p.stem for p in DRIVE_DIR.glob("nb*_*.csv"))
    names = [n for n in names if (RESULT_DIR / f"Result_{n}.csv").exists()]
    print(f"{len(names)} paired samples")

    with ProcessPoolExecutor() as ex:
        out = list(ex.map(load_pair, names, chunksize=8))

    theta, phi = out[0][3], out[0][4]
    for o in out:
        assert np.array_equal(o[3], theta) and np.array_equal(o[4], phi), o[0]
    weights = np.stack([o[1] for o in out])
    gain = np.stack([o[2] for o in out])
    assert not np.isnan(gain).any(), "missing pattern samples"
    n_beams = np.array([int(re.match(r"nb(\d+)_", n).group(1)) for n in names])

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT, weights=weights, gain_db=gain, theta_deg=theta.astype(np.float32),
        phi_deg=phi.astype(np.float32), n_beams=n_beams, names=np.array(names),
    )
    print(f"saved {OUT}: weights {weights.shape}, gain {gain.shape}, "
          f"gain range [{gain.min():.1f}, {gain.max():.1f}] dB")


if __name__ == "__main__":
    main()
