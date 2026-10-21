"""OroLIFT: Orography-modulated Latent Implicit Field Transform.

Design in one line: do all the heavy work on the 32x64 input grid, and spend as
little as possible per output pixel -- then hand the result to a zero-parameter
projection that makes it exactly consistent with the input.

    x (69,32,64) --encoder--> latent (C,32,64) --1x1--> modulations (Cm,32,64)
                                                              |
      HR queries (delta-coords, cell scale, static orography)  |  FiLM
                                \___ tiny shared MLP at 121x240 ___/
                                              |
                             residual --> y = U(x) + s * r,  then project(y, x)

The decoder never runs a wide network at HR: only a 3-layer width-64 MLP whose
weights are shared across all output pixels and whose behavior is set by
latent modulations sampled from the LR grid.  That is what makes the whole thing
scale-free (3.75x is not special) and cheap.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grids import SeparableCoarsener, LAT_HR, LON_HR, LAT_LR, LON_LR


def sphere_pad(x, p):
    """Circular in longitude, replicate in latitude."""
    x = F.pad(x, (p, p, 0, 0), mode="circular")
    return F.pad(x, (0, 0, p, p), mode="replicate")


class SphereConv(nn.Module):
    def __init__(self, ci, co, k=3, groups=1, dilation=1):
        super().__init__()
        self.p = (k // 2) * dilation
        self.c = nn.Conv2d(ci, co, k, padding=0, groups=groups, dilation=dilation)

    def forward(self, x):
        return self.c(sphere_pad(x, self.p) if self.p else x)


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        v = x.var(1, keepdim=True, unbiased=False)
        return (x - m) * torch.rsqrt(v + 1e-6) * self.g + self.b


class SAFM(nn.Module):
    """Spatially-Adaptive Feature Modulation (SAFMN, ICCV'23), sphere-aware.

    Channels are split into `n` groups; group i is pooled to 1/2^i resolution,
    convolved, and pushed back up.  Multi-scale context for the price of one
    depthwise conv per group -- the cheap way to get a large receptive field on a
    32x64 grid where a transformer would be overkill.
    """

    def __init__(self, dim, n=4):
        super().__init__()
        self.n, self.d = n, dim // n
        self.convs = nn.ModuleList([SphereConv(self.d, self.d, 3, groups=self.d) for _ in range(n)])
        self.agg = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        xs = torch.chunk(x, self.n, dim=1)
        out = [self.convs[0](xs[0])]
        for i in range(1, self.n):
            h, w = max(H >> i, 2), max(W >> i, 2)
            z = F.adaptive_max_pool2d(xs[i], (h, w))
            z = self.convs[i](z)
            out.append(F.interpolate(z, (H, W), mode="bilinear", align_corners=False))
        return x * self.act(self.agg(torch.cat(out, 1)))


class CCM(nn.Module):
    def __init__(self, dim, ratio=2):
        super().__init__()
        self.f = nn.Sequential(SphereConv(dim, dim * ratio, 3), nn.GELU(),
                               nn.Conv2d(dim * ratio, dim, 1))

    def forward(self, x):
        return self.f(x)


class Block(nn.Module):
    def __init__(self, dim, n=4, ratio=2):
        super().__init__()
        self.n1, self.safm = LayerNorm2d(dim), SAFM(dim, n)
        self.n2, self.ccm = LayerNorm2d(dim), CCM(dim, ratio)

    def forward(self, x):
        x = x + self.safm(self.n1(x))
        return x + self.ccm(self.n2(x))


class OroLIFT(nn.Module):
    def __init__(self, n_var=69, n_static_lr=5, n_static_hr=7, dim=96, depth=8,
                 render_width=64, render_depth=3, direct_dim=32, mode="conservative",
                 resid_std=None, use_projection=True, use_orography=True):
        super().__init__()
        self.n_var, self.use_projection, self.use_orography = n_var, use_projection, use_orography
        self.C = SeparableCoarsener(mode, dtype=torch.float32)
        for p in self.C.parameters():
            p.requires_grad_(False)

        self.head = SphereConv(n_var + n_static_lr, dim, 3)
        self.body = nn.Sequential(*[Block(dim) for _ in range(depth)])
        self.tail = SphereConv(dim, dim, 3)

        self.W, self.LD = render_width, render_depth
        self.to_mod = nn.Conv2d(dim, 2 * render_width * render_depth, 1)
        self.to_direct = nn.Conv2d(dim, direct_dim, 1)

        n_q = 2 + 2 + (n_static_hr if use_orography else 3) + direct_dim
        self.render_in = nn.Conv2d(n_q, render_width, 1)
        self.render_mid = nn.ModuleList([nn.Conv2d(render_width, render_width, 1)
                                         for _ in range(render_depth - 1)])
        self.render_out = nn.Conv2d(render_width, n_var, 1)
        nn.init.zeros_(self.render_out.weight)
        nn.init.zeros_(self.render_out.bias)

        self.register_buffer("resid_std",
                             torch.ones(n_var) if resid_std is None else torch.as_tensor(resid_std))
        # dtype of the skip path + projection; the consistency error sits at this
        # type's roundoff floor (~1e-6 fp32, ~1e-12 fp64), never at bf16's ~3e-3.
        self.proj_dtype = torch.float32
        self._make_query_grid()

    # ---------------------------------------------------------------- queries
    def _make_query_grid(self, lat_lr=None, lon_lr=None):
        """delta = offset of each HR point inside its LR cell, in [-.5,.5]; plus cell scale.

        Both are functions of the LR *geometry* only.
        """
        import numpy as _np
        lat_lr = LAT_LR if lat_lr is None else _np.asarray(lat_lr, dtype=_np.float64)
        lon_lr = LON_LR if lon_lr is None else _np.asarray(lon_lr, dtype=_np.float64)
        dlat, dlon = lat_lr[1] - lat_lr[0], lon_lr[1] - lon_lr[0]
        flat = (LAT_HR - lat_lr[0]) / dlat
        flon = (LON_HR - lon_lr[0]) / dlon
        dla = torch.as_tensor(flat - flat.round(), dtype=torch.float32)
        dlo = torch.as_tensor(flon - flon.round(), dtype=torch.float32)
        q = torch.stack([dla[:, None].expand(len(LAT_HR), len(LON_HR)),
                         dlo[None, :].expand(len(LAT_HR), len(LON_HR))])
        # cell scale in units of the HR spacing: tells the decoder *how far* it is
        # upsampling (the ratio of input to output spacing).
        s = torch.tensor([dlat / (LAT_HR[1] - LAT_HR[0]), dlon / (LON_HR[1] - LON_HR[0])],
                         dtype=torch.float32)
        s = s[:, None, None].expand(2, len(LAT_HR), len(LON_HR)) / 8.0
        self.register_buffer("query", torch.cat([q, s])[None])

        # grid_sample coordinates for LR -> HR latent lookup.  The latent is padded
        # by one cell (circular in lon, replicate in lat) so that bilinear sampling
        # reproduces the same clamped/periodic interpolation as U_lat (x) U_lon,
        # at ~70x lower cost than the two GEMMs (it only touches 4 taps per pixel).
        fy = torch.as_tensor(flat, dtype=torch.float32).clamp(0, len(lat_lr) - 1) + 1.0
        fx = torch.as_tensor(flon, dtype=torch.float32) + 1.0
        gy = 2.0 * fy / (len(lat_lr) + 1) - 1.0
        gx = 2.0 * fx / (len(lon_lr) + 1) - 1.0
        grid = torch.stack([gx[None, :].expand(len(LAT_HR), len(LON_HR)),
                            gy[:, None].expand(len(LAT_HR), len(LON_HR))], -1)
        self.register_buffer("grid", grid[None])

    def set_lr_grid(self, lat_lr, lon_lr, resid_std=None):
        """Re-target the model to a different input resolution, zero-shot.

        ``resid_std`` is the output scale and is *resolution dependent* -- the
        sub-grid residual at 1.88x is half the size of the one at 3.75x.  Leaving it
        at the training value makes the model over-correct by exactly that factor,
        which shows up first in geopotential because its residual is the smallest.
        """
        dev = self.query.device
        if resid_std is not None:
            self.resid_std = torch.as_tensor(resid_std, dtype=self.resid_std.dtype,
                                             device=dev)
        self.C = SeparableCoarsener(self.C.mode, dtype=torch.float32,
                                    lat_lr=lat_lr, lon_lr=lon_lr).to(dev)
        self._C64 = None
        q, g = self.query, self.grid
        del self._buffers["query"], self._buffers["grid"]
        self._make_query_grid(lat_lr, lon_lr)
        self.query, self.grid = self.query.to(dev), self.grid.to(dev)
        return self

    def sample_to_hr(self, t):
        """Bilinear LR->HR lookup, periodic in longitude, clamped at the poles."""
        t = F.pad(t, (1, 1, 0, 0), mode="circular")
        t = F.pad(t, (0, 0, 1, 1), mode="replicate")
        return F.grid_sample(t, self.grid.expand(t.shape[0], -1, -1, -1),
                             mode="bilinear", align_corners=True)

    def forward(self, x_lr, static_lr, static_hr):
        """x_lr: (B,69,32,64) normalized.  Returns (B,69,121,240) normalized."""
        B = x_lr.shape[0]
        f = self.tail(self.body(self.head(torch.cat([x_lr, static_lr.expand(B, -1, -1, -1)], 1))))

        mod = self.sample_to_hr(self.to_mod(f))                   # (B, 2*W*LD, 121,240)
        direct = self.sample_to_hr(self.to_direct(f))
        sh = static_hr if self.use_orography else static_hr[:, 4:]  # drop oro/lsm channels
        q = torch.cat([self.query.expand(B, -1, -1, -1), sh.expand(B, -1, -1, -1), direct], 1)

        h = self.render_in(q)
        mods = mod.chunk(2 * self.LD, dim=1)
        h = F.gelu(h * (1 + mods[0]) + mods[1])
        for i, lin in enumerate(self.render_mid):
            h = F.gelu(lin(h) * (1 + mods[2 * (i + 1)]) + mods[2 * (i + 1) + 1])
        r = self.render_out(h)

        # The certificate is a floating-point claim, so the skip path and the
        # projection must escape bf16 autocast: bf16 has 8 mantissa bits, which caps
        # ||A y - x|| at ~3e-3 no matter how exact the algebra is.
        with torch.autocast("cuda", enabled=False):
            d = self.proj_dtype
            if d == torch.float32:
                C = self.C
            else:   # rebuild at the target precision: casting fp32 buffers up keeps
                    # their fp32 roundoff and caps the certificate at ~1e-8
                if getattr(self, "_C64", None) is None:
                    self._C64 = SeparableCoarsener(self.C.mode, dtype=d).to(x_lr.device)
                C = self._C64
            x32, r32 = x_lr.to(d), r.to(d)
            # grid_sample is fp32-only; the fp64 path uses the exact U_lat/U_lon matrices
            up = C.upsample(x32) if d == torch.float64 else OroLIFT.sample_to_hr(self, x32)
            y = up + r32 * self.resid_std.to(d)[None, :, None, None]
            return C.project(y, x32) if self.use_projection else y
