# Coherent quadratic-form physics models: weights -> GainTotal dB.
#
# with embedded element patterns EEP and a power-normalisation D (incident, accepted or
# radiated power). EEP_pol approximates the EEPs with R coherent channels.
#
# this is currently done with channel r convolves the weights with a coupling kernel C_r (translation-invariant
# coupling; zero padding makes edges differ automatically), multiplies by a per-element
# complex map A_r (edge / position effects), evaluates the array factor, and is weighted
# by a smooth complex pattern e_pol,r. Every option keeps the model a ratio of quadratic
# forms in w, so any number of beams is handled exactly by construction.
# 

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from experiments.common import N, FastAF
from physics import C0, sphere_weights


def _horizon_knots(theta_deg, n):
    """n theta knots denser around the horizon (90 deg)."""
    th = torch.as_tensor(theta_deg, dtype=torch.float32).cpu()
    u = torch.linspace(0, 1, n)
    y = torch.tanh((u - 0.5) * 3.0)
    y = (y - y[0]) / (y[-1] - y[0])
    return th[0] + y * (th[-1] - th[0])


class ComplexSphereMap(nn.Module):
    #n smooth complex fields on the (theta, phi) grid: Fourier series in phi with
    #coefficients piecewise-linear (or cubic) in theta.

    def __init__(self, theta_deg, phi_deg, n, knots=37, harm=8, init_const=None,
                 knot_mode="uniform"):
        super().__init__()
        self.T = len(theta_deg)
        ph = torch.deg2rad(torch.as_tensor(phi_deg, dtype=torch.float32).cpu())
        m = torch.arange(1, harm + 1, dtype=torch.float32)
        basis = torch.cat([torch.ones(1, len(ph)), torch.cos(m[:, None] * ph),
                           torch.sin(m[:, None] * ph)])
        self.register_buffer("basis", basis)
        self.coef = nn.Parameter(torch.zeros(n, 2, 2 * harm + 1, knots))
        if init_const is not None:
            with torch.no_grad():
                self.coef[:, 0, 0] = torch.as_tensor(init_const)[:, None]
        if knot_mode == "horizon":
            th = torch.as_tensor(theta_deg, dtype=torch.float32).cpu()
            self.register_buffer("Wt", _interp_matrix(th, _horizon_knots(theta_deg, knots)))
        else:
            self.Wt = None

    def forward(self):
        n, _, h, k = self.coef.shape
        flat = self.coef.reshape(n * 2, h, k)
        if self.Wt is None:
            c = F.interpolate(flat, size=self.T, mode="linear", align_corners=True)
        else:
            c = torch.einsum("ahk,tk->aht", flat, self.Wt)
        f = torch.einsum("aht,hp->atp", c, self.basis).reshape(n, 2, self.T, -1)
        return torch.complex(f[:, 0], f[:, 1])


class SplitSphereMap(nn.Module):
    #Independent front / back hemisphere maps

    # Back-lobe physics (finite ground plane, diffraction) is not a smooth
    #continuation of the forward embedded-element pattern.
    

    def __init__(self, theta_deg, phi_deg, n, knots=37, harm=8, init_const=None,
                 knot_mode="uniform"):
        super().__init__()
        self.front = ComplexSphereMap(theta_deg, phi_deg, n, knots, harm, init_const,
                                      knot_mode)
        self.back = ComplexSphereMap(theta_deg, phi_deg, n, knots, harm, None, knot_mode)
        th = torch.as_tensor(theta_deg, dtype=torch.float32).cpu()
        self.register_buffer("blend", torch.sigmoid((th - 90.0) / 4.0))

    def forward(self):
        b = self.blend[None, :, None]
        return self.front() * (1 - b) + self.back() * b


def _interp_matrix(x, knots, periodic=False):
    #  Linear interpolation matrix
    n = len(knots)
    M = torch.zeros(len(x), n)
    r = torch.arange(len(x))
    if periodic:
        step = knots[1] - knots[0]
        pos = (x - knots[0]) / step
        i0 = torch.floor(pos).long()
        fr = pos - i0
        M[r, i0 % n] += 1 - fr
        M[r, (i0 + 1) % n] += fr
        return M
    i1 = torch.searchsorted(knots, x).clamp(1, n - 1)
    i0 = i1 - 1
    span = (knots[i1] - knots[i0]).clamp_min(1e-8)
    fr = ((x - knots[i0]) / span).clamp(0, 1)
    M[r, i0] += 1 - fr
    M[r, i1] += fr
    return M


class LowResEEP(nn.Module):
    #Free per-element complex pattern f_e on a coarse (theta, phi) knot grid:
    # E_q(g) = sum_e w_e exp(jk r_e . u_g) f_q,e(g). The full-resolution EEP tensor
    #[E, T*P] is formed per polarisation (0.5 GB) under checkpointing, then one GEMM

    def __init__(self, af, theta_deg, phi_deg, n_pol, t_knots=37, p_knots=72):
        super().__init__()
        self.af = af
        th = torch.as_tensor(theta_deg, dtype=torch.float32).cpu()
        ph = torch.as_tensor(phi_deg, dtype=torch.float32).cpu()
        self.register_buffer("Wt", _interp_matrix(th, torch.linspace(0, 180, t_knots)))
        pk = ph[0] + torch.arange(p_knots) * 360.0 / p_knots
        self.register_buffer("Wp", _interp_matrix(ph, pk, periodic=True))
        n = af.n
        self.f = nn.Parameter(torch.zeros(n_pol, 2, n * n, t_knots, p_knots))

    def _pol(self, w, f):
        fu = self.Wt @ (f @ self.Wp.t())  # [2,E,T,P]
        eep = torch.complex(fu[0], fu[1]).flatten(1) * self.af.phase_matrix()
        return w @ eep  # [B, T*P]

    def forward(self, w):
        """w complex [B, n, n] -> fields [B, q, T, P]."""
        w = w.flatten(1)
        out = [checkpoint(self._pol, w, self.f[q], use_reentrant=False)
               for q in range(self.f.shape[0])]
        return torch.stack(out, 1).reshape(w.shape[0], -1, self.af.T, self.af.P)


def complex_conv(w, k, ext=0):
    """w [B,N,N] complex, k [R,K,K] complex -> [B,R,N+2ext,N+2ext] (zero padded)."""
    p = k.shape[-1] // 2 + ext
    x = torch.stack([w.real, w.imag], 1)  # [B,2,N,N]
    kr, ki = k.real[:, None], k.imag[:, None]
    wr = torch.cat([torch.cat([kr, -ki], 1), torch.cat([ki, kr], 1)], 0)  # [2R,2,K,K]
    y = F.conv2d(x, wr, padding=p)
    R = k.shape[0]
    return torch.complex(y[:, :R], y[:, R:])


def edge_maps(ext=0):
    n = N + 2 * ext
    i = torch.arange(n, dtype=torch.float32) - ext
    d = torch.stack([i, N - 1 - i]).abs()  # distance to low / high array edge
    rows = torch.exp(-d / 1.5)  # [2,n]
    m = [rows[0][:, None].expand(n, n), rows[1][:, None].expand(n, n),
         rows[0][None].expand(n, n), rows[1][None].expand(n, n)]
    return torch.stack(m)

# Coherent physics model
class CoherentPhysics(nn.Module):
    def __init__(self, theta_deg, phi_deg, n_pol=1, rank=1, kernel=7, elem_map="shared",
                 norm="directivity", floor=True, amp="lin", knots=37, harm=8,
                 norm_kernel=7, init_noise=0.02, f_ghz=2.4, map_init="random", ext=0,
                 eep=None, glob=False, incoh=0, incoh_mode="proj", epat_split=False,
                 image=False, knot_mode="uniform", incoh_harm=None, incoh_knots=None,
                 incoh_split=False):
        super().__init__()
        self.ext = ext
        n = N + 2 * ext
        self.af = FastAF(theta_deg, phi_deg, f_ghz, n=n)
        self.register_buffer("sw", sphere_weights(torch.as_tensor(theta_deg).cpu().float()))
        self.n_pol, self.R, self.K = n_pol, rank, kernel
        self.norm, self.amp, self.use_floor = norm, amp, floor
        self.glob, self.incoh, self.incoh_mode = glob, incoh, incoh_mode
        self.image = image
        g = torch.Generator().manual_seed(0)
        k = torch.zeros(rank, 2, kernel, kernel)
        k[:, 0, kernel // 2, kernel // 2] = 1.0
        k[1:] += init_noise * torch.randn(rank - 1, 2, kernel, kernel, generator=g)
        self.kern = nn.Parameter(k)
        # per-element complex maps: none | shared (one map, all channels) | channel
        self.elem_map = elem_map
        if elem_map != "none":
            nm = 1 if elem_map == "shared" else rank
            a = torch.zeros(nm, 2, n, n)
            a[:, 0] = 1.0
            if nm > 1 and map_init == "edge":
                a[1:min(nm, 5), 0] = edge_maps(ext)[:min(nm, 5) - 1]
            elif nm > 1:
                a[1:] += init_noise * torch.randn(nm - 1, 2, n, n, generator=g)
            self.amap = nn.Parameter(a)
        init = torch.zeros(n_pol * rank)
        init[0] = 1.0
        Map = SplitSphereMap if epat_split else ComplexSphereMap
        self.epat = Map(theta_deg, phi_deg, n_pol * rank, knots, harm, init, knot_mode)
        # eep = [t_knots, p_knots]: add free per-element patterns on the element grid
        self.eep = None
        if eep:
            self.eep_af = self.af if ext == 0 else FastAF(theta_deg, phi_deg, f_ghz, n=N)
            self.eep = LowResEEP(self.eep_af, theta_deg, phi_deg, n_pol, *eep)
        if glob:
            # per-channel global coupling: y += <w, B_r> * M_r
            # both factors start small
            self.gB = nn.Parameter(init_noise * torch.randn(rank, 2, n, n, generator=g))
            self.gM = nn.Parameter(torch.zeros(rank, 2, n, n))
            with torch.no_grad():
                self.gM[:, 0] = 0.01
        if incoh:
            self.iB = nn.Parameter(init_noise * torch.randn(incoh, 2, n, n, generator=g))
            Map = SplitSphereMap if incoh_split else ComplexSphereMap
            self.ipat = Map(theta_deg, phi_deg, incoh, incoh_knots or knots,
                            incoh_harm or harm, torch.full((incoh,), 0.05), knot_mode)
            if incoh_mode == "af":
                ik = torch.zeros(incoh, 2, kernel, kernel)
                ik[:, 0, kernel // 2, kernel // 2] = 1.0
                self.ikern = nn.Parameter(ik)
                ia = torch.zeros(incoh, 2, n, n)
                ia[:, 0] = 1.0
                self.iamap = nn.Parameter(ia)
        if image:
            th = torch.deg2rad(torch.as_tensor(theta_deg, dtype=torch.float32).cpu())
            self.register_buffer("cos_th", torch.cos(th))
            self.k0 = 2 * math.pi * f_ghz * 1e9 / C0
            self.h_m = nn.Parameter(torch.tensor(0.002))
            self.gamma = ComplexSphereMap(theta_deg, phi_deg, 1, knots, harm,
                                          torch.tensor([0.02]), knot_mode)
        if floor:
            self.floor_pat = ComplexSphereMap(theta_deg, phi_deg, 1, knots, harm, None,
                                              knot_mode)
            self.floor_db = nn.Parameter(torch.tensor(-45.0))
        if norm == "directivity":
            self.eff_db = nn.Parameter(torch.tensor(0.0))
        else:
            self.scale_db = nn.Parameter(torch.tensor(0.0))
        if norm == "quad":
            kd = torch.zeros(2, norm_kernel, norm_kernel)
            kd[0, norm_kernel // 2, norm_kernel // 2] = 1.0
            self.nkern = nn.Parameter(kd)

    def prep(self, w):
        w = w / w.abs().flatten(1).amax(1).clamp_min(1e-12)[:, None, None]
        if self.amp == "sqrt":
            w = w / w.abs().clamp_min(1e-12).sqrt()
        return w

    def _aperture(self, w):
        if not self.ext:
            return w
        p = self.ext
        return torch.complex(F.pad(w.real, (p, p, p, p)), F.pad(w.imag, (p, p, p, p)))

    def fields(self, w):
        """Complex fields [B,pol,T,P], reference power D(w) [B], and prepped w."""
        w = self.prep(w)
        y = complex_conv(w, torch.complex(self.kern[:, 0], self.kern[:, 1]), self.ext)
        if self.elem_map != "none":
            y = y * torch.complex(self.amap[:, 0], self.amap[:, 1])[None]
        if self.glob:
            wa = self._aperture(w)
            s = torch.einsum("bhw,rhw->br", wa, torch.complex(self.gB[:, 0], self.gB[:, 1]))
            y = y + s[:, :, None, None] * torch.complex(self.gM[:, 0], self.gM[:, 1])
        af = self.af(y)  # [B,R,T,P]
        e = self.epat().reshape(self.n_pol, self.R, *af.shape[-2:])
        E = torch.einsum("qrtp,brtp->bqtp", e, af)
        if self.eep is not None:
            E = E + self.eep(w)
        if self.image:
            img = self.gamma()[0] * torch.exp(
                1j * (2 * self.k0 * self.h_m * self.cos_th)[:, None])
            E = E * (1 + img)
        if self.norm == "incident":
            dw = w.abs().pow(2).sum((1, 2))
        elif self.norm == "quad":
            yd = complex_conv(w, torch.complex(self.nkern[0], self.nkern[1])[None])
            dw = yd.abs().pow(2).sum((1, 2, 3))
        else:
            dw = None
        return E, dw, w

    def scatter_power(self, w):
        # Per-channel incoherent AF power [B,S,T,P]
        yi = complex_conv(w, torch.complex(self.ikern[:, 0], self.ikern[:, 1]), self.ext)
        yi = yi * torch.complex(self.iamap[:, 0], self.iamap[:, 1])
        return (self.ipat()[None] * self.af(yi)).abs().pow(2)

    def extra_power(self, w):
        # Incoherent (non-interfering) power [B,T,P], or 0.
        if not self.incoh:
            return 0.0
        if self.incoh_mode == "af":
            return self.scatter_power(w).sum(1)
        wa = self._aperture(w)
        s = torch.einsum("bhw,shw->bs", wa, torch.complex(self.iB[:, 0], self.iB[:, 1]))
        return torch.einsum("bs,stp->btp", s.abs().pow(2), self.ipat().abs().pow(2))

    def assemble(self, pc, ps, dw):
        # coherent power + scatter power -> gain dB, matching `gain` to consider incoherent power (v.2)
        p = pc + ps
        if self.use_floor:
            ref = (p.mean(2) * self.sw).sum(1) if dw is None else dw
            fl = 10 ** ((self.floor_db + self.floor_pat()[0].real) / 10)
            p = p + fl[None] * ref[:, None, None]
        if self.norm == "directivity":
            p = p / (p.mean(2) * self.sw).sum(1)[:, None, None]
            return 10 * torch.log10(p.clamp_min(1e-12)) + self.eff_db
        p = p / dw[:, None, None]
        return 10 * torch.log10(p.clamp_min(1e-12)) + self.scale_db

    def gain(self, E, dw, w=None):
        pc = E.abs().pow(2).sum(1)
        extra = self.extra_power(w) if w is not None else 0.0
        ps = extra if torch.is_tensor(extra) else torch.zeros_like(pc)
        return self.assemble(pc, ps, dw)

    def forward(self, w):
        E, dw, wp = self.fields(w)
        return self.gain(E, dw, wp)
