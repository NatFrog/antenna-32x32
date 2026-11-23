#Cheap residual heads: physics pattern (+ optional excitation) -> dB or field correction.

#These exist because the 8M-param U-Net on 600 samples overfits (i.e. I had train 0.25 vs
#val 0.91). Each head has a strong bias matching a remaining error mode.


import torch
import torch.nn as nn
import torch.nn.functional as F

from model import WeightEncoder, DB_FLOOR


class BatchedPhiMap(nn.Module):
    # Fourier-in-phi, linear-in-theta map with batched coefficients [B, harm, knots].

    def __init__(self, theta_deg, phi_deg, knots=37, harm=8):
        super().__init__()
        self.T, self.knots, self.harm = len(theta_deg), knots, harm
        ph = torch.deg2rad(torch.as_tensor(phi_deg, dtype=torch.float32).cpu())
        m = torch.arange(1, harm + 1, dtype=torch.float32)
        basis = torch.cat([torch.ones(1, len(ph)), torch.cos(m[:, None] * ph),
                           torch.sin(m[:, None] * ph)])
        self.register_buffer("basis", basis)
        self.n_coef = (2 * harm + 1) * knots

    def forward(self, coef):
        c = coef.reshape(coef.shape[0], 2 * self.harm + 1, self.knots)
        cT = F.interpolate(c, size=self.T, mode="linear", align_corners=True)
        return torch.einsum("bht,hp->btp", cT, self.basis)


class SphereFiLM(nn.Module):
    # Excitation-conditioned smooth dB residual on the sphere.

    # Tests whether leftover error is a scan-dependent but spatially-smooth
    # pattern (element-pattern / scan-impedance residual), not local lobe wiggling.
    #

    def __init__(self, theta_deg, phi_deg, knots=37, harm=8, cond_dim=256):
        super().__init__()
        self.enc = WeightEncoder(cond_dim)
        self.basis = BatchedPhiMap(theta_deg, phi_deg, knots, harm)
        self.to_coef = nn.Linear(cond_dim, self.basis.n_coef)
        nn.init.zeros_(self.to_coef.weight)
        nn.init.zeros_(self.to_coef.bias)

    def forward(self, phys_db, w):
        return self.basis(self.to_coef(self.enc(w)))


class LocalAngleMLP(nn.Module):
    # Shared 1x1 MLP at each (theta, phi): no spatial mixing, so it cannot move lobes.

    # Tests whether leftover error is a local dB recalibration (floor depth, null fill)
    # given the physics value and direction.

    def __init__(self, theta_deg, phi_deg, hidden=64, cond=True, cond_dim=128):
        super().__init__()
        th = torch.deg2rad(torch.as_tensor(theta_deg, dtype=torch.float32).cpu())
        ph = torch.deg2rad(torch.as_tensor(phi_deg, dtype=torch.float32).cpu())
        T, P = len(th), len(ph)
        coords = torch.stack([
            torch.cos(th)[:, None].expand(T, P), torch.sin(th)[:, None].expand(T, P),
            torch.cos(ph)[None].expand(T, P), torch.sin(ph)[None].expand(T, P)])
        self.register_buffer("coords", coords)
        self.enc = WeightEncoder(cond_dim) if cond else None
        self.c1 = nn.Conv2d(6, hidden, 1)
        self.n1 = nn.GroupNorm(8, hidden)
        self.c2 = nn.Conv2d(hidden, hidden, 1)
        self.c3 = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)
        self.film = nn.Linear(cond_dim, 2 * hidden) if cond else None

    def forward(self, phys_db, w):
        B = phys_db.shape[0]
        x0 = phys_db.clamp_min(DB_FLOOR)[:, None]
        rel = x0 - x0.amax((2, 3), keepdim=True)
        x = torch.cat([x0 / 20, rel / 20, self.coords.expand(B, -1, -1, -1)], 1)
        h = self.n1(F.silu(self.c1(x)))
        if self.film is not None:
            s, b = self.film(self.enc(w))[..., None, None].chunk(2, 1)
            h = h * (1 + s) + b
        return self.c3(F.silu(self.c2(h)))[:, 0]


class FieldMul(nn.Module):
    """Complex multiplicative correction of the physics field: E <- E * (1 + eps * d(w)).

    Residual lives in the field (interference is preserved) rather than in dB.
    """

    def __init__(self, theta_deg, phi_deg, knots=37, harm=8, cond_dim=256, scale=0.1):
        super().__init__()
        self.enc = WeightEncoder(cond_dim)
        self.basis = BatchedPhiMap(theta_deg, phi_deg, knots, harm)
        self.to_coef = nn.Linear(cond_dim, 2 * self.basis.n_coef)
        nn.init.zeros_(self.to_coef.weight)
        nn.init.zeros_(self.to_coef.bias)
        self.scale = scale

    def forward(self, w):
        c = self.to_coef(self.enc(w))
        n = self.basis.n_coef
        re, im = self.basis(c[:, :n]), self.basis(c[:, n:])
        return self.scale * torch.complex(re, im)


class ScatterScaleUNet(nn.Module):
    # dB correction of the scatter power only, from a low-pass U-Net.

    # The deep floor is the incoherent AF term. Multiplying that power leaves the
    # coherent beams alone, and dropping skip connections keeps the correction too
    # smooth to memorise a 181x360 speckle map of 600 excitations.
    # 

    def __init__(self, theta_deg, phi_deg, ch=(8, 16, 32), bound=3.0):
        super().__init__()
        from model import ResidualUNet
        self.net = ResidualUNet(theta_deg, phi_deg, ch=ch, n_extra=2, skips=False,
                                bound=bound)

    def forward(self, phys_db, w, ps, pc, dw):
        scale = dw[:, None, None].clamp_min(1e-12)
        extra = torch.stack([
            10 * torch.log10((ps / scale).clamp_min(1e-12)) / 20,
            10 * torch.log10((pc / scale).clamp_min(1e-12)) / 20], 1)
        return self.net(phys_db, w, extra)


class ChannelScale(nn.Module):
    # Per-excitation dB gain of each scatter channel. No spatial mixing.

    # Tests whether the leftover floor is the wrong mix of the existing scatter
    # patterns rather than angular structure those patterns cannot represent.
    # 

    def __init__(self, n, cond_dim=128, bound=6.0):
        super().__init__()
        self.enc = WeightEncoder(cond_dim)
        self.fc = nn.Linear(cond_dim, n)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.bound = bound

    def forward(self, w):
        return self.bound * torch.tanh(self.fc(self.enc(w)))


def build_residual(res, theta_deg, phi_deg, unet_ch=None, cond=True, n_incoh=0):
    if res == "film":
        return SphereFiLM(theta_deg, phi_deg)
    if res == "localmlp":
        return LocalAngleMLP(theta_deg, phi_deg, cond=cond)
    if res == "fieldmul":
        return FieldMul(theta_deg, phi_deg)
    if res == "scatter":
        return ScatterScaleUNet(theta_deg, phi_deg, ch=tuple(unet_ch or (8, 16, 32)))
    if res == "chscale":
        return ChannelScale(n_incoh)
    from model import ResidualUNet
    net = ResidualUNet(theta_deg, phi_deg, ch=unet_ch or (32, 64, 128, 256))
    if not cond:
        class _NoCond(nn.Module):
            def forward(self, w):
                return w.real.new_zeros(w.shape[0], 256)
        net.enc = _NoCond()
    return net
