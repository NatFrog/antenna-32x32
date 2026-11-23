# Plot predictions vs simulation for held-out samples of a trained run.


import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
from metrics import find_beams
from model import DB_FLOOR, HybridModel


def load_model(run: Path, d, dev="cuda"):
    args = json.loads((run / "args.json").read_text())
    model = HybridModel(d.theta_deg, d.phi_deg,
                        unet_ch=tuple(int(c) for c in args["unet_ch"].split(","))).to(dev)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    return model.eval()


def plot_sample(ax_row, name, tgt, phys, hyb, th, ph, nb):
    ext = [ph[0], ph[-1], th[-1], th[0]]
    vmax = float(np.ceil(tgt.max() / 5) * 5)
    for a, img, title in zip(ax_row[:3], (tgt, phys, hyb),
                             ("simulation", "physics stage", "hybrid model")):
        im = a.imshow(np.maximum(img, DB_FLOOR), extent=ext, aspect="auto", cmap="jet",
                      vmin=vmax - 50, vmax=vmax)
        a.set_title(f"{name}: {title}")
    plt.colorbar(im, ax=ax_row[2], label="gain dBi")
    err = np.maximum(hyb, DB_FLOOR) - np.maximum(tgt, DB_FLOOR)
    im = ax_row[3].imshow(err, extent=ext, aspect="auto", cmap="RdBu_r", vmin=-10, vmax=10)
    ax_row[3].set_title("hybrid - simulation (dB)")
    plt.colorbar(im, ax=ax_row[3])
    # theta cut through each true beam
    _, _, idx = find_beams(torch.tensor(tgt), torch.tensor(th), torch.tensor(ph), nb)
    a = ax_row[4]
    for n, i in enumerate(idx.tolist()):
        j = i % len(ph)
        c = f"C{n}"
        a.plot(th, tgt[:, j], c, lw=1.6, label=f"sim phi={ph[j]:.1f}")
        a.plot(th, hyb[:, j], c, ls="--", lw=1.2, label="hybrid")
    a.set_ylim(vmax - 60, vmax + 2)
    a.set_xlabel("theta (deg)")
    a.set_title("theta cuts through beams")
    a.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--per-group", type=int, default=2)
    args = ap.parse_args()
    run = Path(args.run)
    d = D.load().to("cuda")
    model = load_model(run, d)
    sp = np.load(run / "split.npz")
    th, ph = d.theta_deg.cpu().numpy(), d.phi_deg.cpu().numpy()
    out = run / "plots"
    out.mkdir(exist_ok=True)

    sets = {"test": sp["test"]}
    if len(sp["unseen"]):
        sets["unseen"] = sp["unseen"]
    rng = np.random.default_rng(0)
    for sname, idx in sets.items():
        picks = []
        for k in np.unique(d.n_beams[idx]):
            cand = idx[d.n_beams[idx] == k]
            picks += rng.choice(cand, min(args.per_group, len(cand)), replace=False).tolist()
        fig, ax = plt.subplots(len(picks), 5, figsize=(28, 3.6 * len(picks)))
        ax = np.atleast_2d(ax)
        with torch.no_grad():
            for r, i in enumerate(picks):
                hyb, phys = model(d.weights[i:i + 1])
                plot_sample(ax[r], str(d.names[i]), d.gain_db[i].cpu().numpy(),
                            phys[0].cpu().numpy(), hyb[0].cpu().numpy(), th, ph,
                            int(d.n_beams[i]))
        plt.tight_layout()
        f = out / f"{sname}_examples.png"
        plt.savefig(f, dpi=60)
        plt.close(fig)
        print("saved", f)

    rep = run / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        print(f"\n{'set/model':22s} {'group':6s} {'rmse_sph':>9s} {'rmse_top30':>11s} "
              f"{'rmse_top10':>11s} {'beam_gain':>10s} {'beam_pt':>8s}")
        for key, groups in r.items():
            for g, m in groups.items():
                print(f"{key:22s} {g:6s} {m['rmse_sphere_db']:9.2f} {m['rmse_top30db']:11.2f} "
                      f"{m['rmse_top10db']:11.2f} {m['beam_gain_err_db']:10.2f} "
                      f"{m['beam_point_err_deg']:8.2f}")


if __name__ == "__main__":
    main()
