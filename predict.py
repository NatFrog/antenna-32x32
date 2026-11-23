# Predict the radiation pattern for one or more drive-table CSVs.

# Writes CSVs in the simulator's format (Theta[deg],Phi[deg],dB(GainTotal)), easier

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data as D
from evaluate import load_model
from physics import N


def read_drive_table(path) -> np.ndarray:
    d = pd.read_csv(path)
    w = np.zeros((N, N), np.complex64)
    w[d.row_i.values, d.col_j.values] = d.weight_re.values + 1j * d.weight_im.values
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--physics-only", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = D.load()
    model = load_model(Path(args.run), d.to(dev), dev)
    th, ph = d.theta_deg.cpu().numpy(), d.phi_deg.cpu().numpy()
    tt, pp = np.meshgrid(th, ph, indexing="ij")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for t in args.tables:
        w = torch.from_numpy(read_drive_table(t))[None].to(dev)
        with torch.no_grad():
            hyb, phys = model(w)
        g = (phys if args.physics_only else hyb)[0].cpu().numpy()
        # simulator order: phi outer, theta inner
        df = pd.DataFrame({"Theta[deg]": tt.T.ravel(), "Phi[deg]": pp.T.ravel(),
                           "dB(GainTotal)": np.round(g.T.ravel(), 4)})
        f = out / f"Pred_{Path(t).stem}.csv"
        df.to_csv(f, index=False)
        print(f"{f}: peak {g.max():.2f} dBi")


if __name__ == "__main__":
    main()
