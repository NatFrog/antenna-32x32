# Loss and evaluation metrics for gain patterns in dB on a (theta, phi) grid.

import torch
import torch.nn.functional as F

from model import DB_FLOOR


def pixel_weights(target_db: torch.Tensor, theta_deg: torch.Tensor,
                  main_window_db=30.0, main_boost=3.0) -> torch.Tensor:
    """Solid-angle weighting plus extra emphasis on the region within
    `main_window_db` of each pattern's peak (beams and near sidelobes)."""
    s = torch.sin(torch.deg2rad(theta_deg))[None, :, None] + 0.05
    pk = target_db.amax((1, 2), keepdim=True)
    m = (target_db > pk - main_window_db).float()
    w = s * (1 + main_boost * m)
    return w / w.mean((1, 2), keepdim=True)


def loss_fn(pred_db, target_db, theta_deg):
    p = pred_db.clamp_min(DB_FLOOR)
    t = target_db.clamp_min(DB_FLOOR)
    w = pixel_weights(target_db, theta_deg)
    return (w * (p - t).abs()).mean()


def _unit_vec(theta_deg, phi_deg):
    t, p = torch.deg2rad(theta_deg), torch.deg2rad(phi_deg)
    return torch.stack([torch.sin(t) * torch.cos(p), torch.sin(t) * torch.sin(p), torch.cos(t)], -1)


def _ang_deg(a, b):
    return torch.rad2deg(torch.arccos((a * b).sum(-1).clamp(-1, 1)))


def find_beams(x_db, theta_deg, phi_deg, k, nms=9, min_sep_deg=4.0):
    # Top-k local maxima of a single pattern [T,P] (front hemisphere). Returns
    # (values [k], unit vectors [k,3], flat indices [k]). **Check against first idea
    T, P = x_db.shape
    x = x_db[None, None]
    xp = F.pad(x, (nms // 2, nms // 2, 0, 0), mode="circular")
    xp = F.pad(xp, (0, 0, nms // 2, nms // 2), mode="replicate")
    is_max = (x == F.max_pool2d(xp, nms, 1))[0, 0]
    is_max &= (theta_deg < 90)[:, None]
    vals = torch.where(is_max, x_db, torch.full_like(x_db, -1e9)).flatten()
    order = vals.argsort(descending=True)
    th = theta_deg[:, None].expand(T, P).flatten()
    ph = phi_deg[None].expand(T, P).flatten()
    picked = []
    for i in order[:200].tolist():
        if vals[i] < -1e8:
            break
        v = _unit_vec(th[i], ph[i])
        if all(_ang_deg(v, _unit_vec(th[j], ph[j])) > min_sep_deg for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    idx = torch.tensor(picked, dtype=torch.long, device=x_db.device)
    return vals[idx], _unit_vec(th[idx], ph[idx]), idx


@torch.no_grad()
def pattern_metrics(pred_db, target_db, theta_deg, phi_deg, n_beams):
    """Per-sample metrics dict for a batch."""
    p = pred_db.clamp_min(DB_FLOOR)
    t = target_db.clamp_min(DB_FLOOR)
    sw = torch.sin(torch.deg2rad(theta_deg))[None, :, None].expand_as(t)
    e2 = (p - t) ** 2
    out = {
        "rmse_sphere_db": ((e2 * sw).sum((1, 2)) / sw.sum((1, 2))).sqrt(),
        "mae_sphere_db": (((p - t).abs() * sw).sum((1, 2)) / sw.sum((1, 2))),
    }
    pk = t.amax((1, 2), keepdim=True)
    for win in (10, 30):
        m = (t > pk - win).float()
        out[f"rmse_top{win}db"] = ((e2 * m).sum((1, 2)) / m.sum((1, 2))).sqrt()
    out["peak_gain_err_db"] = (p.amax((1, 2)) - t.amax((1, 2))).abs()

    beam_gain, beam_point = [], []
    T, P = t.shape[1:]
    for b in range(t.shape[0]):
        k = int(n_beams[b])
        tv, tu, ti = find_beams(t[b], theta_deg, phi_deg, k)
        pv, pu, _ = find_beams(p[b], theta_deg, phi_deg, k)
        # gain error at the true beam directions
        beam_gain.append((p[b].flatten()[ti] - tv).abs().mean())
        # pointing error: each true beam to nearest predicted beam
        if len(pu):
            d = _ang_deg(tu[:, None], pu[None]).amin(1)
            beam_point.append(d.mean())
        else:
            beam_point.append(torch.tensor(180.0, device=t.device))
    out["beam_gain_err_db"] = torch.stack(beam_gain)
    out["beam_point_err_deg"] = torch.stack(beam_point)
    return out
