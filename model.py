# Hybrid physics + neural model: excitation weights -> gain pattern dB

# Stage 1 (PhysicsModel): learnable mutual-coupling kernel and edge gains act on the
# weights, the exact array factor is evaluated on the sphere, multiplied by a learnable
# smooth embedded-element pattern, plus a learnable incoherent floor, then normalised to
# directivity and offset by a learnable efficiency. Linear in the weights, so it handles
# any number of simultaneous beams by construction.

# Stage 2 (ResidualUNet): a U-Net on the (theta, phi) map, circular in phi, predicts a dB
# correction to the physics pattern, conditioned on an encoding of the weight map.
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics import N, ArrayFactor, sphere_weights

DB_FLOOR = -60.0


class SmoothSphereMap(nn.Module):
    # Smooth scalar field on the (theta, phi) grid: Fourier series in phi whose
    # coefficients are piecewise-linear in theta. Exactly periodic in phi.

    def __init__(self, theta_deg, phi_deg, n_theta_knots=37, n_harm=8):
        super().__init__()
        self.T, self.P = len(theta_deg), len(phi_deg)
        self.coef = nn.Parameter(torch.zeros(1, 2 * n_harm + 1, n_theta_knots))
        ph = torch.deg2rad(torch.as_tensor(phi_deg, dtype=torch.float32).cpu())
        m = torch.arange(1, n_harm + 1, dtype=torch.float32)
        basis = torch.cat([torch.ones(1, self.P), torch.cos(m[:, None] * ph),
                           torch.sin(m[:, None] * ph)])  # [2M+1,P]
        self.register_buffer("basis", basis)

    def forward(self) -> torch.Tensor:
        c = F.interpolate(self.coef, size=self.T, mode="linear", align_corners=True)[0]
        return c.t() @ self.basis  # [T,P]


class PhysicsModel(nn.Module):
    def __init__(self, theta_deg, phi_deg, f_ghz=2.4, coupling_k=7):
        super().__init__()
        self.af = ArrayFactor(theta_deg, phi_deg, f_ghz=f_ghz)
        self.register_buffer("sw", sphere_weights(torch.as_tensor(theta_deg).cpu().float()))
        self.k = coupling_k
        # complex coupling kernel, initialised to identity (delta)
        self.coup = nn.Parameter(torch.zeros(2, coupling_k, coupling_k))
        # per-element complex gain correction (edge / embedded-element effects)
        self.elem_gain = nn.Parameter(torch.zeros(2, N, N))
        self.elem_pat = SmoothSphereMap(theta_deg, phi_deg)  # dB
        self.floor_pat = SmoothSphereMap(theta_deg, phi_deg)  # dB rel. to floor_db
        self.floor_db = nn.Parameter(torch.tensor(-45.0))
        self.eff_db = nn.Parameter(torch.tensor(0.0))

    def effective_weights(self, w: torch.Tensor) -> torch.Tensor:
        p = self.k // 2
        delta = torch.zeros_like(self.coup[0])
        delta[p, p] = 1.0
        kr, ki = (self.coup[0] + delta)[None, None], self.coup[1][None, None]
        x = torch.view_as_real(w)  # [B,N,N,2]
        xr, xi = x[..., 0][:, None], x[..., 1][:, None]
        yr = F.conv2d(xr, kr, padding=p) - F.conv2d(xi, ki, padding=p)
        yi = F.conv2d(xr, ki, padding=p) + F.conv2d(xi, kr, padding=p)
        y = torch.complex(yr[:, 0], yi[:, 0])
        g = torch.complex(1 + self.elem_gain[0], self.elem_gain[1])
        return y * g

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Returns gain in dBi, [B,T,P].
        w = w / w.abs().flatten(1).amax(1).clamp_min(1e-12)[:, None, None]
        we = self.effective_weights(w)
        p_af = self.af(we).abs().pow(2)
        ep = 10 ** (self.elem_pat() / 10)
        fl = 10 ** ((self.floor_db + self.floor_pat()) / 10)
        coherent_norm = (p_af.mean(2) * self.sw).sum(1)  # [B]
        p = ep * p_af + fl * coherent_norm[:, None, None]
        p = p / (p.mean(2) * self.sw).sum(1)[:, None, None]
        return 10 * torch.log10(p.clamp_min(1e-12)) + self.eff_db


def _pad(x, p=1):
    x = F.pad(x, (p, p, 0, 0), mode="circular")  # phi is periodic
    return F.pad(x, (0, 0, p, p), mode="replicate")


class Conv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3)

    def forward(self, x):
        return self.conv(_pad(x))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim=0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = Conv(cin, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = Conv(cout, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.film = nn.Linear(cond_dim, 2 * cout) if cond_dim else None

    def forward(self, x, cond=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.film is not None:
            s, b = self.film(cond)[..., None, None].chunk(2, 1)
            h = h * (1 + s) + b
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class WeightEncoder(nn.Module):
    # 32x32 complex weight map -> conditioning vector (global-phase normalised).

    def __init__(self, dim=256):
        super().__init__()
        c = 64
        self.net = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 4 * c), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * c, dim), nn.SiLU(),
        )

    def forward(self, w):
        w = w / w.abs().flatten(1).amax(1).clamp_min(1e-12)[:, None, None]
        ref = w[:, N // 2, N // 2]
        w = w * (ref.conj() / ref.abs().clamp_min(1e-12))[:, None, None]
        x = torch.stack([w.real, w.imag, w.abs(),
                         20 * torch.log10(w.abs().clamp_min(1e-4)) / 40], 1)
        return self.net(x)


class ResidualUNet(nn.Module):
    def __init__(self, theta_deg, phi_deg, ch=(32, 64, 128, 256), cond_dim=256,
                 n_extra=0, skips=True, bound=0.0):
        super().__init__()
        th = torch.deg2rad(torch.as_tensor(theta_deg, dtype=torch.float32).cpu())
        ph = torch.deg2rad(torch.as_tensor(phi_deg, dtype=torch.float32).cpu())
        T, P = len(th), len(ph)
        coords = torch.stack([
            torch.cos(th)[:, None].expand(T, P), torch.sin(th)[:, None].expand(T, P),
            torch.cos(ph)[None].expand(T, P), torch.sin(ph)[None].expand(T, P)])
        self.register_buffer("coords", coords)
        self.T = T
        self.Tp = math.ceil(T / 2 ** (len(ch) - 1)) * 2 ** (len(ch) - 1)
        self.skips, self.bound = skips, bound
        self.enc = WeightEncoder(cond_dim)
        self.inp = nn.Conv2d(2 + 4 + n_extra, ch[0], 1)
        self.down = nn.ModuleList()
        cin = ch[0]
        for c in ch:
            self.down.append(nn.ModuleList([ResBlock(cin, c, cond_dim), ResBlock(c, c, cond_dim)]))
            cin = c
        self.mid = nn.ModuleList([ResBlock(cin, cin, cond_dim), ResBlock(cin, cin, cond_dim)])
        self.up = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.up.append(nn.ModuleList([
                ResBlock(cin + (c if skips else 0), c, cond_dim), ResBlock(c, c, cond_dim)]))
            cin = c
        self.out_norm = nn.GroupNorm(8, cin)
        self.out = nn.Conv2d(cin, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, phys_db, w, extra=None):
        B = phys_db.shape[0]
        cond = self.enc(w)
        x0 = phys_db.clamp_min(DB_FLOOR)[:, None]
        rel = x0 - x0.amax((2, 3), keepdim=True)
        parts = [x0 / 20, rel / 20, self.coords.expand(B, -1, -1, -1)]
        if extra is not None:
            parts.append(extra)
        x = torch.cat(parts, 1)
        x = F.pad(x, (0, 0, 0, self.Tp - self.T), mode="replicate")
        h = self.inp(x)
        skips = []
        for i, (b1, b2) in enumerate(self.down):
            h = b2(b1(h, cond), cond)
            if i < len(self.down) - 1:
                if self.skips:
                    skips.append(h)
                h = F.avg_pool2d(h, 2)
        for b in self.mid:
            h = b(h, cond)
        for b1, b2 in self.up:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            if self.skips:
                h = torch.cat([h, skips.pop()], 1)
            h = b2(b1(h, cond), cond)
        d = self.out(F.silu(self.out_norm(h)))[:, 0, :self.T]
        if self.bound:
            d = self.bound * torch.tanh(d)
        return d.float()


class HybridModel(nn.Module):
    def __init__(self, theta_deg, phi_deg, unet_ch=(32, 64, 128, 256)):
        super().__init__()
        self.physics = PhysicsModel(theta_deg, phi_deg)
        self.residual = ResidualUNet(theta_deg, phi_deg, ch=unet_ch)

    def forward(self, w, use_residual=True, amp=True):
        phys = self.physics(w)
        if not use_residual:
            return phys, phys
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and w.is_cuda):
            delta = self.residual(phys.detach(), w)
        return phys + delta, phys
