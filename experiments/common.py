# Shared experiment infrastructure: data, array factor, eval, run dirs

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data as D  # noqa: E402
from metrics import loss_fn, pattern_metrics  # noqa: E402
from physics import C0, N, PITCH_MM  # noqa: E402

RUNS = ROOT / "runs" / "exp"


class _AFFn(torch.autograd.Function):
    """AF = sum_mn w_mn ex[g,m] ey[g,n]; backward is the adjoint, so the [B,G,N]
    intermediate is never stored."""

    @staticmethod
    def forward(ctx, w, ex, ey):
        ctx.save_for_backward(ex, ey)
        return (torch.matmul(ex, w) * ey).sum(-1)

    @staticmethod
    def backward(ctx, g):
        ex, ey = ctx.saved_tensors
        gw = torch.matmul((ex.conj() * g[..., None]).transpose(-1, -2), ey.conj())
        return gw, None, None


class FastAF(torch.nn.Module):
    def __init__(self, theta_deg, phi_deg, f_ghz=2.4, n=N):
        super().__init__()
        th = torch.deg2rad(torch.as_tensor(theta_deg, dtype=torch.float32).cpu())
        ph = torch.deg2rad(torch.as_tensor(phi_deg, dtype=torch.float32).cpu())
        u = (torch.sin(th)[:, None] * torch.cos(ph)[None]).flatten()
        v = (torch.sin(th)[:, None] * torch.sin(ph)[None]).flatten()
        pos = (torch.arange(n, dtype=torch.float32) - (n - 1) / 2) * PITCH_MM * 1e-3
        self.n = n
        k = 2 * math.pi * f_ghz * 1e9 / C0
        self.register_buffer("ex", torch.exp(1j * k * u[:, None] * pos).to(torch.complex64))
        self.register_buffer("ey", torch.exp(1j * k * v[:, None] * pos).to(torch.complex64))
        self.T, self.P = len(th), len(ph)
        self._pm = None

    def phase_matrix(self):
        """[n*n, T*P] element phase matrix, built lazily and not saved in checkpoints."""
        if self._pm is None or self._pm.device != self.ex.device:
            self._pm = (self.ex[:, :, None] * self.ey[:, None, :]).reshape(
                -1, self.n * self.n).t().contiguous()
        return self._pm

    def forward(self, w):
        """w complex [..., N, N] -> AF complex [..., T, P]."""
        sh = w.shape[:-2]
        af = w.reshape(-1, self.n * self.n) @ self.phase_matrix()
        return af.reshape(*sh, self.T, self.P)


def get_data(dev="cuda"):
    d = D.load().to(dev)
    tr, va, te = D.split(d.n_beams, seed=0)
    return d, tr, va, te


@torch.no_grad()
def evaluate(model, d, idx, bs=8, metrics=True):
    was = model.training
    model.eval()
    losses, rows = [], []
    for i in range(0, len(idx), bs):
        b = torch.as_tensor(idx[i:i + bs], device=d.weights.device)
        pred = model(d.weights[b])
        tgt = d.gain_db[b]
        losses.append(loss_fn(pred, tgt, d.theta_deg).item() * len(b))
        if metrics:
            m = pattern_metrics(pred, tgt, d.theta_deg, d.phi_deg, d.n_beams[idx[i:i + bs]])
            for j in range(len(b)):
                rows.append({k: v[j].item() for k, v in m.items()}
                            | {"n_beams": int(d.n_beams[idx[i + j]])})
    model.train(was)
    return float(np.sum(losses) / len(idx)), rows


def summarize(rows):
    keys = [k for k in rows[0] if k != "n_beams"]
    groups = {"all": rows} | {f"nb{k}": [r for r in rows if r["n_beams"] == k]
                              for k in sorted({r["n_beams"] for r in rows})}
    return {g: {k: float(np.mean([r[k] for r in rs])) for k in keys} | {"n": len(rs)}
            for g, rs in groups.items()}


def fmt(s):
    a = s["all"]
    return (f"rmse_sph {a['rmse_sphere_db']:.2f} | top30 {a['rmse_top30db']:.3f} | "
            f"top10 {a['rmse_top10db']:.3f} | bgain {a['beam_gain_err_db']:.3f} | "
            f"bpoint {a['beam_point_err_deg']:.3f}")


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, *a):
        s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")


def run_dir(cfg):
    p = RUNS / f"{cfg['id']}_{cfg['name']}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))
