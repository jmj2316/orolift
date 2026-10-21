"""Heteroscedastic head: a second, independent estimate of how much sub-grid
structure the coarse state leaves undetermined.

The flow head measures that quantity as ensemble spread.  This one measures it
directly, by fitting a per-gridpoint conditional standard deviation of the
deterministic model's residual under a Gaussian likelihood.  The two share no
machinery beyond the frozen mean model, so agreement between them is evidence that
the number is a property of the problem rather than of one generative model.

Both estimate Var[y | x] *given this mean model*.  If the deterministic mean is not the
true conditional mean, both are upper bounds on the irreducible part.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .orolift import SphereConv, LayerNorm2d


class ResBlockD(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.n = LayerNorm2d(dim)
        self.c1 = SphereConv(dim, dim, 3, dilation=dilation)
        self.c2 = SphereConv(dim, dim, 3, dilation=dilation)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(self.n(x))))


class VarHead(nn.Module):
    """log sigma(x) at 1.5 deg, in units of the free-baseline residual scale.

    Output 0 means "as uncertain as the whole baseline residual"; negative means the
    deterministic model has pinned that location down.
    """

    def __init__(self, n_var=69, n_static_hr=7, dim=48, depth=4,
                 dilations=(1, 2, 4, 1), lo=-6.0, hi=3.0):
        super().__init__()
        self.lo, self.hi = lo, hi
        self.inp = SphereConv(n_var + n_static_hr, dim, 3)
        d = list(dilations) + [1] * max(0, depth - len(dilations))
        self.blocks = nn.ModuleList([ResBlockD(dim, d[i]) for i in range(depth)])
        self.out = SphereConv(dim, n_var, 3)
        nn.init.zeros_(self.out.c.weight); nn.init.zeros_(self.out.c.bias)

    def forward(self, y_det_scaled, static_hr):
        h = self.inp(torch.cat([y_det_scaled,
                                static_hr.expand(y_det_scaled.shape[0], -1, -1, -1)], 1))
        for b in self.blocks:
            h = b(h)
        return self.out(h).clamp(self.lo, self.hi)          # log sigma


def gaussian_nll(e_scaled, log_sigma, w):
    """Lat-weighted Gaussian NLL of the scaled residual, per variable."""
    inv2 = torch.exp(-2.0 * log_sigma)
    return (w * (0.5 * e_scaled ** 2 * inv2 + log_sigma)).mean(dim=(-2, -1))
