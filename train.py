# Train the hybrid physics + U-Net radiation-pattern model.

# Stage A fits the low-parameter physics model (coupling, element pattern, floor).
# Stage B trains the residual U-Net jointly, with a time-based cosine LR schedule so a
# fixed wall-clock budget is used fully.

# i.e. python train.py --hours 1 --train-beams 1,2 --out runs/holdout_nb3
# 

import argparse
import copy
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import data as D
from metrics import loss_fn, pattern_metrics
from model import HybridModel


def flip_phi_index(P, axis):
    i = np.arange(P)
    # phi grid is -179.5..179.5 in 1 deg steps
    return (179 - i) % P if axis == "x" else (P - 1 - i)


def augment(w, g, flip):
    if not flip:
        return w, g
    P = g.shape[2]
    if torch.rand(()) < 0.5:  # x -> -x  <=>  phi -> 180 - phi
        w = w.flip(1)
        g = g[:, :, torch.as_tensor(flip_phi_index(P, "x"), device=g.device)]
    if torch.rand(()) < 0.5:  # y -> -y  <=>  phi -> -phi
        w = w.flip(2)
        g = g[:, :, torch.as_tensor(flip_phi_index(P, "y"), device=g.device)]
    return w, g


@torch.no_grad()
def evaluate(model, d, idx, use_residual=True, bs=8):
    model.eval()
    th, ph = d.theta_deg, d.phi_deg
    rows, losses = [], []
    for i in range(0, len(idx), bs):
        b = torch.as_tensor(idx[i:i + bs])
        pred, _ = model(d.weights[b.to(d.weights.device)], use_residual=use_residual)
        tgt = d.gain_db[b.to(d.gain_db.device)]
        losses.append(loss_fn(pred, tgt, th).item() * len(b))
        m = pattern_metrics(pred, tgt, th, ph, d.n_beams[idx[i:i + bs]])
        for j in range(len(b)):
            rows.append({k: v[j].item() for k, v in m.items()} | {
                "name": str(d.names[idx[i + j]]), "n_beams": int(d.n_beams[idx[i + j]])})
    model.train()
    return float(np.sum(losses) / len(idx)), rows


def summarize(rows):
    keys = [k for k in rows[0] if k not in ("name", "n_beams")]
    out = {}
    groups = {"all": rows} | {f"nb{k}": [r for r in rows if r["n_beams"] == k]
                              for k in sorted({r["n_beams"] for r in rows})}
    for gname, rs in groups.items():
        out[gname] = {k: float(np.mean([r[k] for r in rs])) for k in keys} | {"n": len(rs)}
    return out


def fmt(s):
    a = s["all"]
    return (f"rmse_sph {a['rmse_sphere_db']:.2f} | rmse_top30 {a['rmse_top30db']:.2f} | "
            f"rmse_top10 {a['rmse_top10db']:.2f} | beam_gain {a['beam_gain_err_db']:.2f} dB | "
            f"beam_point {a['beam_point_err_deg']:.2f} deg")


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
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--hours", type=float, default=7.0, help="stage-B wall-clock budget")
    ap.add_argument("--physics-steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--physics-lr", type=float, default=2e-2)
    ap.add_argument("--unet-ch", default="32,64,128,256")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every-min", type=float, default=10.0)
    ap.add_argument("--train-beams", default="1,2,3",
                    help="beam counts used for training; others are evaluated as unseen")
    ap.add_argument("--flip-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if sys.platform == "win32":  # keep the machine awake while training
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    log = open(out / "log.txt", "a", buffering=1)

    def say(*a):
        s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")

    d = D.load().to(dev)
    tr, va, te = D.split(d.n_beams, seed=0)
    seen = [int(x) for x in args.train_beams.split(",")]
    unseen = np.flatnonzero(~np.isin(d.n_beams, seen))
    tr = tr[np.isin(d.n_beams[tr], seen)]
    va = va[np.isin(d.n_beams[va], seen)]
    np.savez(out / "split.npz", train=tr, val=va, test=te, unseen=unseen)
    say(f"train {len(tr)} val {len(va)} test {len(te)} unseen-beam-count {len(unseen)}")

    model = HybridModel(d.theta_deg, d.phi_deg,
                        unet_ch=tuple(int(c) for c in args.unet_ch.split(","))).to(dev)
    say(f"physics params {sum(p.numel() for p in model.physics.parameters())}, "
        f"unet params {sum(p.numel() for p in model.residual.parameters()) / 1e6:.2f}M")

    # ---------------- Stage A: physics model ----------------
    ckpt_a = out / "physics.pt"
    if ckpt_a.exists():
        model.physics.load_state_dict(torch.load(ckpt_a))
        say("loaded existing physics stage")
    else:
        opt = torch.optim.Adam(model.physics.parameters(), lr=args.physics_lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, args.physics_lr, total_steps=args.physics_steps, pct_start=0.05)
        t0 = time.time()
        for step in range(args.physics_steps):
            b = torch.as_tensor(np.random.choice(tr, 16, replace=False), device=dev)
            w, g = augment(d.weights[b], d.gain_db[b], args.flip_aug)
            pred, _ = model(w, use_residual=False)
            loss = loss_fn(pred, g, d.theta_deg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == args.physics_steps - 1:
                say(f"[physics] step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
        torch.save(model.physics.state_dict(), ckpt_a)
    vl, rows = evaluate(model, d, va, use_residual=False)
    say(f"[physics] val loss {vl:.3f} | {fmt(summarize(rows))}")

    if args.hours <= 0:
        return

    # ---------------- Stage B: joint with residual U-Net ----------------
    opt = torch.optim.AdamW([
        {"params": model.residual.parameters(), "lr": args.lr, "weight_decay": 1e-4},
        {"params": model.physics.parameters(), "lr": args.lr * 0.25, "weight_decay": 0.0},
    ])
    base_lrs = [g["lr"] for g in opt.param_groups]
    ema = EMA(model, args.ema)
    budget = args.hours * 3600
    warm = 300.0  # seconds of linear warmup...
    t0 = time.time()
    last_eval = t0
    best = math.inf
    step = 0
    hist = []
    while True:
        el = time.time() - t0
        if el > budget:
            break
        frac = el / budget
        scale = min(1.0, el / warm) * 0.5 * (1 + math.cos(math.pi * frac))
        for g_, lr0 in zip(opt.param_groups, base_lrs):
            g_["lr"] = lr0 * max(scale, 0.01)

        perm = np.random.permutation(tr)
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            b = torch.as_tensor(perm[i:i + args.batch], device=dev)
            w, g = augment(d.weights[b], d.gain_db[b], args.flip_aug)
            pred, _ = model(w)
            loss = loss_fn(pred, g, d.theta_deg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            step += 1
            hist.append(loss.item())
            # history of a heir of my mother in my house of my land and 
            # i will stand a poet, i gotta study, and add another option

        if time.time() - last_eval > args.eval_every_min * 60 or time.time() - t0 > budget:
            last_eval = time.time()
            vl, rows = evaluate(ema.model, d, va)
            s = summarize(rows)
            tag = ""
            if vl < best:
                best = vl
                torch.save(ema.model.state_dict(), out / "best.pt")
                tag = " *"
            say(f"[joint] {el / 3600:.2f}h step {step} lr {opt.param_groups[0]['lr']:.2e} "
                f"train {np.mean(hist[-200:]):.3f} val {vl:.3f}{tag} | {fmt(s)}")
            torch.save(ema.model.state_dict(), out / "last.pt")

    # ---------------- Final evaluation ----------------
    model.load_state_dict(torch.load(out / "best.pt"))
    report = {}
    for name, idx in (("val", va), ("test", te), ("unseen", unseen)):
        if len(idx) == 0:
            continue
        for mode, res in (("physics", False), ("hybrid", True)):
            _, rows = evaluate(model, d, idx, use_residual=res)
            report[f"{name}/{mode}"] = summarize(rows)
            say(f"[final] {name:6s} {mode:7s} | {fmt(report[f'{name}/{mode}'])}")
            if mode == "hybrid":
                with open(out / f"{name}_per_sample.json", "w") as f:
                    json.dump(rows, f, indent=1)
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
