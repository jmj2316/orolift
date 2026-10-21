"""Conditional flow matching for the irreducible part of the downscaling problem,
constrained to the null space of the coarsening operator.

The deterministic model already returns a field with A y_det = x, and the truth
satisfies A y = x too.  Their difference therefore lies in ker(A).  If the velocity
field of a flow is projected into ker(A) at every step, and the initial noise is
too, then *every* ensemble member satisfies A y = x exactly -- the ensemble explores
only the directions the coarse input genuinely leaves undetermined.

That is what turns the spread into a measurement: it is the conditional uncertainty
of the fine scales given everything the coarse state says, with no leakage into
directions the coarse state already fixed.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .orolift import SphereConv, LayerNorm2d


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, dim, tdim, dilation=1):
        super().__init__()
        self.n = LayerNorm2d(dim)
        self.c1 = SphereConv(dim, dim, 3, dilation=dilation)
        self.c2 = SphereConv(dim, dim, 3, dilation=dilation)
        self.mod = nn.Linear(tdim, 2 * dim)

    def forward(self, x, temb):
        g, b = self.mod(temb)[:, :, None, None].chunk(2, dim=1)
        h = self.c1(self.n(x) * (1 + g) + b)
        return x + self.c2(F.gelu(h))


class FlowHead(nn.Module):
    """Velocity field v_theta(e_t, t | y_det, statics), output projected into ker(A).

    Dilations widen the receptive field without extra cost -- the residual it has to
    model is short-wavelength by construction, but its *statistics* depend on the
    synoptic situation, which arrives through y_det rather than through a huge kernel.
    """

    def __init__(self, n_var=69, n_static_hr=7, dim=64, depth=6, tdim=128,
                 dilations=(1, 2, 4, 1, 2, 4), resid_std=None, sigma_cond=False):
        super().__init__()
        self.n_var = n_var
        self.sigma_cond = sigma_cond
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.tdim = tdim
        # Conditioning on a cheap, already-calibrated estimate of the local scale.
        # Without it the head has to infer how uncertain each gridpoint is from the
        # smoothed mean field alone, which is what it was getting wrong: aggregate
        # spread was right (0.987) while its spatial placement correlated only 0.31
        # with the realized error.
        nin = (3 if sigma_cond else 2) * n_var + n_static_hr
        self.inp = SphereConv(nin, dim, 3)
        d = list(dilations) + [1] * max(0, depth - len(dilations))
        self.blocks = nn.ModuleList([FiLMBlock(dim, tdim, d[i]) for i in range(depth)])
        self.out = SphereConv(dim, n_var, 3)
        nn.init.zeros_(self.out.c.weight); nn.init.zeros_(self.out.c.bias)
        self.register_buffer("resid_std",
                             torch.ones(n_var) if resid_std is None else torch.as_tensor(resid_std))

    def forward(self, e_t, t, y_det, static_hr, log_sigma=None):
        temb = self.temb(timestep_embedding(t, self.tdim))
        parts = [e_t, y_det]
        if self.sigma_cond:
            assert log_sigma is not None, "sigma_cond=True needs log_sigma"
            parts.append(log_sigma)
        parts.append(static_hr.expand(e_t.shape[0], -1, -1, -1))
        h = self.inp(torch.cat(parts, 1))
        for blk in self.blocks:
            h = blk(h, temb)
        return self.out(h)


def ker_project(C, e):
    """Orthogonal (area-weighted) projection onto ker(A): P(e, 0)."""
    return C.project(e, torch.zeros_like(C.coarsen(e)))


class FlowSampler:
    """Deterministic Heun integration of the ker(A)-constrained probability flow."""

    def __init__(self, net, C, det_model, var_head=None):
        self.net, self.C, self.det, self.var = net, C, det_model, var_head

    @torch.no_grad()
    def sample(self, x, sl, sh, n_members=8, steps=16, generator=None):
        with torch.autocast("cuda", torch.bfloat16):
            y_det = self.det(x, sl, sh).float()
        ls = None
        if getattr(self.net, "sigma_cond", False):
            rs0 = self.net.resid_std[None, :, None, None]
            with torch.autocast("cuda", torch.bfloat16):
                ls = self.var(y_det / rs0, sh).float()
        B = x.shape[0]
        rs = self.net.resid_std[None, :, None, None]
        out = []
        for _ in range(n_members):
            g = torch.randn(y_det.shape, device=y_det.device, generator=generator)
            e = ker_project(self.C, g)
            ts = torch.linspace(0, 1, steps + 1, device=y_det.device)
            for i in range(steps):
                t0, t1 = ts[i], ts[i + 1]
                h = t1 - t0
                with torch.autocast("cuda", torch.bfloat16):
                    v0 = self.net(e, t0.repeat(B), y_det / rs, sh, ls).float()
                v0 = ker_project(self.C, v0)
                e1 = e + h * v0
                with torch.autocast("cuda", torch.bfloat16):
                    v1 = self.net(e1, t1.repeat(B), y_det / rs, sh, ls).float()
                v1 = ker_project(self.C, v1)
                e = e + h * 0.5 * (v0 + v1)
            out.append(y_det + e * rs)
        return torch.stack(out, 1)          # (B, M, V, H, W)
