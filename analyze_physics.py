"""Measure how well physics alone (array factor -> directivity + mean element pattern)
explains the simulated gain patterns. Frequency/sign were fitted earlier: 2.4 GHz, +1.
"""

import numpy as np
import torch

from physics import ArrayFactor, directivity_db

data = np.load("data/dataset.npz")
dev = "cuda" if torch.cuda.is_available() else "cpu"
idx = np.arange(0, 750, 5)
w = torch.tensor(data["weights"][idx]).to(dev)
g = torch.tensor(data["gain_db"][idx]).to(dev)
nb = data["n_beams"][idx]
th = torch.tensor(data["theta_deg"]).to(dev)
ph = torch.tensor(data["phi_deg"]).to(dev)

af = ArrayFactor(th, ph, f_ghz=2.4).to(dev)
d = torch.cat([directivity_db(af(w[i:i + 25]), th) for i in range(0, len(w), 25)])

d = d.clamp_min(-40)
g = g.clamp_min(-40)
perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))
w, g, d, nb = w[perm], g[perm], d[perm], nb[perm.numpy()]
half = len(idx) // 2
elem = (g[:half] - d[:half]).median(0).values  # robust element pattern from first half
print("element pattern (gain - AF directivity), mean over phi, every 10 deg theta:")
print(np.round(elem.mean(1)[::10].cpu().numpy(), 2))
print("element pattern std over phi at theta=0,30,60,90:",
      [round(elem[t].std().item(), 2) for t in (0, 30, 60, 90)])

pred = d[half:] + elem
gt = g[half:]
err = pred - gt
nbh = nb[half:]
for k in (1, 2, 3):
    sel = torch.tensor(nbh == k, device=dev)
    e, gg = err[sel], gt[sel]
    m = gg > gg.amax((1, 2), keepdim=True) - 30
    print(f"nb={k}: RMSE all {e.pow(2).mean().sqrt():.2f} dB | within 30 dB of peak "
          f"{e[m].pow(2).mean().sqrt():.2f} dB | peak gain err "
          f"{(pred[sel].amax((1,2)) - gg.amax((1,2))).abs().mean():.2f} dB")

# peak location error in degrees (great-circle)
def peak_dir(x):
    i = x.flatten(1).argmax(1)
    t, p = th[i // x.shape[2]], ph[i % x.shape[2]]
    t, p = torch.deg2rad(t), torch.deg2rad(p)
    return torch.stack([torch.sin(t) * torch.cos(p), torch.sin(t) * torch.sin(p), torch.cos(t)], 1)

ang = torch.rad2deg(torch.arccos((peak_dir(pred) * peak_dir(gt)).sum(1).clamp(-1, 1)))
print("peak direction error deg: median", ang.median().item(), "max", ang.max().item())
