# Run one or more registered experiments sequentially.

#     python -m experiments.run P01 P02          # can run configs by id
#     python -m experiments.run P01 --force      # rerun even if result.json exists

# Each run writes runs/exp/*id*_config.json, log.txt, curve.csv, best.pt, result.json}.
# Training stops at the step budget, the wall-clock cap, or when val loss has not improved
# for patience` evaluations (testing convergence)


import argparse
import copy
import ctypes
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.common import (Logger, evaluate, fmt, get_data, run_dir, save_json,
                                summarize, RUNS)
from experiments.configs import CONFIGS
from experiments.losses import LOSSES
from experiments.phys_models import CoherentPhysics
from experiments.res_models import build_residual


class Hybrid(nn.Module):
    def __init__(self, phys, theta_deg, phi_deg, res="unet", unet_ch=None, cond=True):
        super().__init__()
        self.physics = phys
        self.res_type = res
        self.residual = build_residual(res, theta_deg, phi_deg, unet_ch, cond,
                                       n_incoh=getattr(phys, "incoh", 0))

    def forward(self, w):
        if self.res_type == "fieldmul":
            E, dw, wp = self.physics.fields(w)
            d = self.residual(w)
            return self.physics.gain(E * (1 + d[:, None]), dw, wp)
        if self.res_type == "scatter":
            E, dw, wp = self.physics.fields(w)
            pc = E.abs().pow(2).sum(1)
            ps = self.physics.extra_power(wp)
            base = self.physics.assemble(pc, ps, dw)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=w.is_cuda):
                delta = self.residual(base.detach(), w, ps.detach(), pc.detach(), dw.detach())
            ps = ps * torch.exp(delta.float() * (math.log(10) / 10))
            return self.physics.assemble(pc, ps, dw)
        if self.res_type == "chscale":
            E, dw, wp = self.physics.fields(w)
            pc = E.abs().pow(2).sum(1)
            sp = self.physics.scatter_power(wp)
            delta = self.residual(w)
            sp = sp * torch.exp(delta[:, :, None, None] * (math.log(10) / 10))
            return self.physics.assemble(pc, sp.sum(1), dw)
        p = self.physics(w)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=w.is_cuda):
            delta = self.residual(p.detach(), w)
        return p + delta


def _fit_ipat(old, new_shape):
    # Copy DC / cos / sin blocks into a wider harmonic axis. Higher harmonics stay 0,
    # so a warm-started scatter pattern reproduces the checkpoint exactly.
    if old.shape == new_shape:
        return old
    if old.dim() != 4 or tuple(old.shape[:2]) != tuple(new_shape[:2]) or old.shape[3] != new_shape[3]:
        return None
    h_old = (old.shape[2] - 1) // 2
    h_new = (new_shape[2] - 1) // 2
    if h_new < h_old or old.shape[2] != 2 * h_old + 1:
        return None
    out = old.new_zeros(new_shape)
    out[:, :, 0] = old[:, :, 0]
    out[:, :, 1:1 + h_old] = old[:, :, 1:1 + h_old]
    out[:, :, 1 + h_new:1 + h_new + h_old] = old[:, :, 1 + h_old:]
    return out


def _expand_dim0(old, new_shape):
    # Copy overlapping leading channels
    if old.shape == new_shape:
        return old
    if old.dim() != len(new_shape) or tuple(old.shape[1:]) != tuple(new_shape[1:]):
        return None
    out = old.new_zeros(new_shape)
    n = min(old.shape[0], new_shape[0])
    out[:n] = old[:n]
    return out


def _fit_knots(old, new_shape):
    # Linearly resample the theta-knot axis (last dim of sphere-map coefs).
    if old.shape == new_shape:
        return old
    if old.dim() != 4 or tuple(old.shape[:3]) != tuple(new_shape[:3]):
        return None
    x = old.reshape(-1, 1, old.shape[3])
    y = torch.nn.functional.interpolate(x, size=new_shape[3], mode="linear",
                                        align_corners=True)
    return y.reshape(new_shape)


def _adapt_tensor(old, new_shape):
    # Harmonic pad, knot resample, and/or channel expand as needed.
    if old.shape == new_shape:
        return old
    if old.dim() != 4:
        return _expand_dim0(old, new_shape)
    # align channels first so harm/knot helpers see matching leading dims.
    if old.shape[0] != new_shape[0]:
        mid = (new_shape[0], *old.shape[1:])
        old = _expand_dim0(old, mid)
        if old is None:
            return None
        if old.shape == new_shape:
            return old
    fit = _fit_ipat(old, new_shape)
    if fit is not None:
        return fit
    # harmonics differ and knots differ: pad harm on old knots, then resample knots.
    if old.shape[2] != new_shape[2] and old.shape[3] != new_shape[3]:
        mid = (*new_shape[:3], old.shape[3])
        harm = _fit_ipat(old, mid)
        if harm is None:
            return None
        return _fit_knots(harm, new_shape)
    return _fit_knots(old, new_shape)


def _load_phys(phys, sd, src):
    own = phys.state_dict()
    param_keys = {n for n, _ in phys.named_parameters()}
    merged, adapted, skipped = {}, [], []
    for k, v in own.items():
        merged[k] = v
    for k, v in sd.items():
        # Geometry / Fourier bases are rebuilt for the new architecture
        if k not in param_keys:
            continue
        # Independent front/back copies reproduce the single smooth pattern at step 0.
        if (k == "ipat.coef" and "ipat.front.coef" in own
                and ("ipat.front.coef" not in sd)):
            tgt = own["ipat.front.coef"].shape
            fit = _adapt_tensor(v, tgt)
            if fit is not None:
                merged["ipat.front.coef"] = fit
                merged["ipat.back.coef"] = fit.clone()
                adapted.append(f"ipat.coef -> front/back {tuple(v.shape)} -> {tuple(tgt)}")
                continue
        if k not in own:
            skipped.append(k)
            continue
        if own[k].shape == v.shape:
            merged[k] = v
            continue
        if "ipat" in k and k.endswith(".coef"):
            fit = _adapt_tensor(v, own[k].shape)
            if fit is not None:
                merged[k] = fit
                adapted.append(f"{k} {tuple(v.shape)} -> {tuple(own[k].shape)}")
                continue
        fit = _expand_dim0(v, own[k].shape)
        if fit is not None:
            merged[k] = fit
            adapted.append(f"{k} channels {v.shape[0]} -> {own[k].shape[0]}")
            continue
        skipped.append(f"{k} {tuple(v.shape)} != {tuple(own[k].shape)}")
    phys.load_state_dict(merged)
    if adapted or skipped:
        print(f"load {src}: adapted {adapted or '-'} | skipped {skipped or '-'}", flush=True)


def build(cfg, d):
    m = dict(cfg["model"])
    kind = m.pop("type")
    init = m.pop("init_from", None)
    unet_ch = m.pop("unet_ch", None)
    cond = m.pop("cond", True)
    res = m.pop("res", "unet")
    phys = CoherentPhysics(d.theta_deg, d.phi_deg, **m)
    if init:
        src = next(RUNS.glob(f"{init}_*")) / "best.pt"
        sd = torch.load(src, map_location="cpu")
        sd = {k.removeprefix("physics."): v for k, v in sd.items()
              if not k.startswith("residual.")}
        _load_phys(phys, sd, init)
    if kind == "phys":
        return phys
    return Hybrid(phys, d.theta_deg, d.phi_deg, res, unet_ch, cond)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p, 1 - self.decay)


TRAIN_DEFAULTS = dict(steps=3000, batch=8, lr=2e-2, warmup=0.03, eval_every=100,
                      patience=8, min_delta=1e-3, max_min=30, train_beams=[1, 2, 3],
                      loss="l1db", phys_lr_scale=0.25, wd=1e-4, ema=0.0, seed=0,
                      final_lr_frac=0.01, eep_lr_scale=0.1, min_steps_frac=0.3,
                      aug=False, freeze_phys=False)


def _aug(w, g):
    """D2 flips of the square array (x -> -x and y -> -y) with matching phi maps."""
    P = g.shape[-1]
    if torch.rand((), device=w.device) < 0.5:
        w = w.flip(-2)
        g = g[..., (179 - torch.arange(P, device=g.device)) % P]
    if torch.rand((), device=w.device) < 0.5:
        w = w.flip(-1)
        g = g.flip(-1)
    return w, g


def run(cfg, force=False):
    out = run_dir(cfg)
    if (out / "result.json").exists() and not force:
        print(f"skip {cfg['id']} (done)")
        return
    for f in ("log.txt", "curve.csv"):
        (out / f).unlink(missing_ok=True)
    t = TRAIN_DEFAULTS | cfg.get("train", {})
    cfg = cfg | {"train": t}
    save_json(out / "config.json", cfg)
    say = Logger(out / "log.txt")
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    rng = np.random.default_rng(t["seed"])

    d, tr, va, te = get_data()
    seen = t["train_beams"]
    unseen = np.flatnonzero(~np.isin(d.n_beams, seen))
    tr = tr[np.isin(d.n_beams[tr], seen)]
    va = va[np.isin(d.n_beams[va], seen)]
    model = build(cfg, d).cuda()
    n_par = sum(p.numel() for p in model.parameters())
    say(f"{cfg['id']} {cfg['name']}: {cfg.get('desc', '')}")
    say(f"params {n_par:,} | train {len(tr)} val {len(va)} test {len(te)} unseen {len(unseen)}")

    loss_f = LOSSES[t["loss"]]
    if isinstance(model, Hybrid):
        groups = [{"params": model.residual.parameters(), "lr": t["lr"], "weight_decay": t["wd"]}]
        if not t["freeze_phys"]:
            groups.append({"params": model.physics.parameters(),
                           "lr": t["lr"] * t["phys_lr_scale"], "weight_decay": 0.0})
        else:
            for p in model.physics.parameters():
                p.requires_grad_(False)
        opt = torch.optim.AdamW(groups)
    else:
        eep = [p for n, p in model.named_parameters() if n.startswith("eep.")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("eep.")]
        opt = torch.optim.Adam([{"params": rest, "lr": t["lr"]}]
                               + ([{"params": eep, "lr": t["lr"] * t["eep_lr_scale"]}]
                                  if eep else []))
    base = [g["lr"] for g in opt.param_groups]
    ema = EMA(model, t["ema"]) if t["ema"] > 0 else None

    def lr_at(s):
        w = max(1, int(t["warmup"] * t["steps"]))
        if s < w:
            return (s + 1) / w
        f = (s - w) / max(1, t["steps"] - w)
        lo = t["final_lr_frac"]
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * f))

    curve = open(out / "curve.csv", "w", buffering=1)
    curve.write("step,minutes,lr,train_loss,val_loss\n")
    best, best_step, bad, hist = math.inf, 0, 0, []
    t0 = time.time()
    stop = "steps"
    perm, pi = rng.permutation(tr), 0
    for step in range(t["steps"]):
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * lr_at(step)
        if pi + t["batch"] > len(perm):
            perm, pi = rng.permutation(tr), 0
        b = torch.as_tensor(perm[pi:pi + t["batch"]], device="cuda")
        pi += t["batch"]
        w, g = d.weights[b], d.gain_db[b]
        if t["aug"]:
            w, g = _aug(w, g)
        pred = model(w)
        loss = loss_f(pred, g, d.theta_deg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema:
            ema.update(model)
        hist.append(loss.item())
        last = step == t["steps"] - 1
        mins = (time.time() - t0) / 60
        if (step + 1) % t["eval_every"] == 0 or last or mins > t["max_min"]:
            em = ema.model if ema else model
            vl, _ = evaluate(em, d, va, metrics=False)
            tl = float(np.mean(hist[-t["eval_every"]:]))
            curve.write(f"{step + 1},{mins:.2f},{opt.param_groups[0]['lr']:.3e},{tl:.4f},{vl:.4f}\n")
            tag = ""
            if vl < best * (1 - t["min_delta"]):
                best, best_step, bad, tag = vl, step + 1, 0, " *"
                torch.save(em.state_dict(), out / "best.pt")
            else:
                bad += 1
            say(f"step {step + 1} {mins:.1f}m lr {opt.param_groups[0]['lr']:.2e} "
                f"train {tl:.4f} val {vl:.4f}{tag}")
            if mins > t["max_min"]:
                stop = "time"
                break
            if bad >= t["patience"] and step + 1 >= t["min_steps_frac"] * t["steps"]:
                stop = "converged"
                break

    model.load_state_dict(torch.load(out / "best.pt"))
    res = {"id": cfg["id"], "name": cfg["name"], "desc": cfg.get("desc", ""), "params": n_par,
           "best_step": best_step, "steps_run": step + 1, "stop": stop,
           "minutes": (time.time() - t0) / 60, "val_loss": best}
    sets = [("val", va), ("test", te)] + ([("unseen", unseen)] if len(unseen) else [])
    for name, idx in sets:
        vl, rows = evaluate(model, d, idx)
        s = summarize(rows)
        res[name] = s | {"loss": vl}
        say(f"[final] {name:6s} loss {vl:.4f} | {fmt(s)}")
    save_json(out / "result.json", res)
    say(f"done: stop={stop} best_step={best_step} {res['minutes']:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for i in a.ids:
        cfg = CONFIGS[i]
        try:
            run(cfg, a.force)
        except Exception as e:  # keep the queue going
            import traceback
            traceback.print_exc()
            (run_dir(cfg) / "error.txt").write_text(traceback.format_exc())
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
