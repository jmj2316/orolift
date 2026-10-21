"""Comparison models, all sharing OroLIFT's interface, residual parameterization,
static inputs and (optional) consistency projection -- so the only thing that
differs is *where the computation happens*.

Two families:

* pre-upsampling CNNs (`edsr`, `safmn`, `rcan`) -- the standard climate-downscaling
  recipe: interpolate to the target grid first, then run the whole network at
  121x240.  Handles a non-integer factor trivially, and pays for it on every pixel.
* implicit decoders (`liif`) -- encoder at LR, per-pixel MLP at HR
  fed with the concatenated latent, which is what OroLIFT replaces with FiLM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grids import SeparableCoarsener
from .orolift import OroLIFT, SphereConv, Block, sphere_pad


class _Base(nn.Module):
    """Shared plumbing: LR->HR lookup, residual scaling, consistency projection."""

    def __init__(self, n_var, mode, resid_std, use_projection, use_orography):
        super().__init__()
        proto = OroLIFT(n_var=n_var, mode=mode, resid_std=resid_std,
                        use_projection=use_projection, use_orography=use_orography,
                        dim=8, depth=1, render_width=8, render_depth=2, direct_dim=4)
        self.C = proto.C
        self.register_buffer("grid", proto.grid)
        self.register_buffer("query", proto.query)
        self.register_buffer("resid_std", proto.resid_std)
        self.use_projection, self.use_orography, self.n_var = use_projection, use_orography, n_var
        self.proj_dtype = torch.float32

    sample_to_hr = OroLIFT.sample_to_hr

    def finish(self, r, x_lr):
        # see OroLIFT.forward: the projection must not run in bf16
        with torch.autocast("cuda", enabled=False):
            d = getattr(self, "proj_dtype", torch.float32)
            if d == torch.float32:
                C = self.C
            else:   # rebuild at the target precision: casting fp32 buffers up keeps
                    # their fp32 roundoff and caps the certificate at ~1e-8
                if getattr(self, "_C64", None) is None:
                    self._C64 = SeparableCoarsener(self.C.mode, dtype=d).to(x_lr.device)
                C = self._C64
            x32, r32 = x_lr.to(d), r.to(d)
            # grid_sample is fp32-only; the fp64 path uses the exact U_lat/U_lon matrices
            up = C.upsample(x32) if d == torch.float64 else self.sample_to_hr(x32)
            y = up + r32 * self.resid_std.to(d)[None, :, None, None]
            return C.project(y, x32) if self.use_projection else y


class ResBlock(nn.Module):
    def __init__(self, c, res_scale=0.1):
        super().__init__()
        self.c1, self.c2, self.s = SphereConv(c, c, 3), SphereConv(c, c, 3), res_scale

    def forward(self, x):
        return x + self.s * self.c2(F.relu(self.c1(x)))


class ChannelAttn(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1),
                               nn.ReLU(True), nn.Conv2d(c // r, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class RCAB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(SphereConv(c, c, 3), nn.ReLU(True), SphereConv(c, c, 3),
                               ChannelAttn(c))

    def forward(self, x):
        return x + self.b(x)


class PreUpCNN(_Base):
    """Interpolate to 121x240, then run the whole trunk there."""

    def __init__(self, arch="edsr", n_var=69, n_static_hr=7, dim=64, depth=8,
                 mode="conservative", resid_std=None, use_projection=True,
                 use_orography=True, **kw):
        super().__init__(n_var, mode, resid_std, use_projection, use_orography)
        nin = n_var + (n_static_hr if use_orography else 3)
        self.head = SphereConv(nin, dim, 3)
        blk = {"edsr": ResBlock, "rcan": RCAB, "safmn": Block}[arch]
        self.body = nn.Sequential(*[blk(dim) for _ in range(depth)])
        self.tail = SphereConv(dim, n_var, 3)
        nn.init.zeros_(self.tail.c.weight); nn.init.zeros_(self.tail.c.bias)

    def forward(self, x_lr, static_lr, static_hr):
        B = x_lr.shape[0]
        up = self.sample_to_hr(x_lr)
        sh = static_hr if self.use_orography else static_hr[:, 4:]
        h = self.head(torch.cat([up, sh.expand(B, -1, -1, -1)], 1))
        return self.finish(self.tail(self.body(h) + h), x_lr)


class LIIF(_Base):
    """Local Implicit Image Function: LR encoder, per-pixel MLP on the *concatenated*
    latent (unfolded 3x3 neighbourhood), relative coordinate and cell size."""

    def __init__(self, n_var=69, n_static_lr=5, n_static_hr=7, dim=64, depth=8,
                 hidden=256, mlp_depth=4, mode="conservative", resid_std=None,
                 use_projection=True, use_orography=True, **kw):
        super().__init__(n_var, mode, resid_std, use_projection, use_orography)
        self.head = SphereConv(n_var + n_static_lr, dim, 3)
        self.body = nn.Sequential(*[ResBlock(dim) for _ in range(depth)])
        nin = dim * 9 + 4 + (n_static_hr if use_orography else 3)
        layers, c = [], nin
        for _ in range(mlp_depth - 1):
            layers += [nn.Conv2d(c, hidden, 1), nn.ReLU(True)]
            c = hidden
        self.mlp = nn.Sequential(*layers, nn.Conv2d(c, n_var, 1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_lr, static_lr, static_hr):
        B = x_lr.shape[0]
        f = self.body(self.head(torch.cat([x_lr, static_lr.expand(B, -1, -1, -1)], 1)))
        f = F.unfold(sphere_pad(f, 1), 3).view(B, -1, *x_lr.shape[-2:])
        sh = static_hr if self.use_orography else static_hr[:, 4:]
        q = torch.cat([self.sample_to_hr(f), self.query.expand(B, -1, -1, -1),
                       sh.expand(B, -1, -1, -1)], 1)
        return self.finish(self.mlp(q), x_lr)


def build_model(arch, **kw):
    if arch == "orolift":
        return OroLIFT(**kw)
    if arch == "liif":
        return LIIF(**{k: v for k, v in kw.items() if k not in ("render_width", "render_depth", "direct_dim")})
    if arch in ("edsr", "rcan", "safmn"):
        return PreUpCNN(arch=arch, **{k: v for k, v in kw.items()
                                      if k not in ("render_width", "render_depth", "direct_dim")})
    raise ValueError(arch)
