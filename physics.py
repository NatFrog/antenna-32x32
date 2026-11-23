"""Differentiable array-factor computation for the 32x32 planar array."""

import math

import torch
import torch.nn as nn

N = 32
PITCH_MM = 62.5
C0 = 299_792_458.0


class ArrayFactor(nn.Module):
    """AF(theta, phi) = sum_mn w_mn exp(j*sign*k*(x_m*u + y_n*v)) on a fixed grid.

    Computed separably: AF = Ex @ W @ Ey^T per grid point, which is cheap on GPU.
    """

    def __init__(self, theta_deg, phi_deg, f_ghz: float, sign: int = 1,
                 swap_xy: bool = False):
        super().__init__()
        th = torch.as_tensor(theta_deg, dtype=torch.float32).cpu() * math.pi / 180
        ph = torch.as_tensor(phi_deg, dtype=torch.float32).cpu() * math.pi / 180
        u = torch.sin(th)[:, None] * torch.cos(ph)[None, :]  # [T,P]
        v = torch.sin(th)[:, None] * torch.sin(ph)[None, :]
        if swap_xy:
            u, v = v, u
        pos = (torch.arange(N, dtype=torch.float32) - (N - 1) / 2) * PITCH_MM * 1e-3
        k = 2 * math.pi * f_ghz * 1e9 / C0
        ex = torch.exp(1j * sign * k * u[..., None] * pos)  # [T,P,N] for rows (x)
        ey = torch.exp(1j * sign * k * v[..., None] * pos)  # [T,P,N] for cols (y)
        self.register_buffer("ex", ex.to(torch.complex64))
        self.register_buffer("ey", ey.to(torch.complex64))
        self.T, self.P = u.shape

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: complex [B,32,32] -> complex AF [B,T,P]."""
        ex = self.ex.reshape(-1, N)  # [TP,N]
        ey = self.ey.reshape(-1, N)
        t = torch.einsum("gm,bmn->bgn", ex, w)  # [B,TP,N]
        af = (t * ey[None]).sum(-1)
        return af.reshape(w.shape[0], self.T, self.P)

    def power_db(self, w: torch.Tensor, floor_db: float = -100.0) -> torch.Tensor:
        p = self.forward(w).abs().pow(2)
        return 10 * torch.log10(p.clamp_min(10 ** (floor_db / 10)))


def sphere_weights(theta_deg: torch.Tensor) -> torch.Tensor:
    """Quadrature weights over theta (sin theta), normalised so sum*P = 1 per grid."""
    s = torch.sin(torch.deg2rad(theta_deg)).clamp_min(1e-4)
    return s / s.sum()


def directivity_db(af: torch.Tensor, theta_deg: torch.Tensor,
                   floor_db: float = -100.0) -> torch.Tensor:
    """Isotropic-element directivity of a complex AF [B,T,P] in dBi."""
    p = af.abs().pow(2)
    sw = sphere_weights(theta_deg)
    mean_p = (p.mean(2) * sw).sum(1)  # [B]
    d = p / mean_p[:, None, None]
    return 10 * torch.log10(d.clamp_min(10 ** (floor_db / 10)))
