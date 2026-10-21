"""Separable lat-lon grid operators for OroLIFT.

The whole point of this module: the coarsening operator that maps a 1.5-degree
field to a 5.625-degree field is *separable*,

    A vec(Y) = vec(A_lat @ Y @ A_lon.T),      A = A_lat (x) A_lon

so every quantity that would normally need a (32*64) x (121*240) matrix collapses
to two small factors.  In particular the exact projection onto the affine set
{Y : A Y = X} costs two 32x32 / 64x64 solves instead of a 2048x2048 one, and it
is exact for *any* separable linear coarsener -- conservative or interpolating,
integer or non-integer scale factor, cos(phi)-weighted or not.
"""

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
def exact_matmul():
    """Run einsums at true fp32.

    The certificate is only as good as the arithmetic under it, and three separate
    global settings can silently cap it: bf16 autocast (~3e-3), TF32 matmul (~2e-4,
    and it is on by default in most training scripts), and EMA-averaging the
    operator's own constant buffers.  The operator therefore pins its own precision
    rather than trusting the caller's flags.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

# ---------------------------------------------------------------- grids

# HR: WeatherBench2 1.5 deg, pole-inclusive point grid.
LAT_HR = np.linspace(-90.0, 90.0, 121)
LON_HR = np.arange(240) * 1.5

# LR: WeatherBench 5.625 deg, cell-centered.  NOTE: the lat.npy that ships with
# some 5.625-degree HDF5 stores claims linspace(-90, 90, 32); that file is wrong -- the stored fields live on
# this cell-centered grid (verified: Z500 mismatch 288 -> 44 gpm).
LAT_LR = -87.1875 + 5.625 * np.arange(32)
LON_LR = np.arange(64) * 5.625

R_EARTH = 6371.0e3


def cell_edges(centers, periodic=False, period=360.0, clip=None):
    """Midpoint cell edges for a 1-D coordinate of cell centers."""
    c = np.asarray(centers, dtype=np.float64)
    if periodic:
        ext = np.concatenate([[c[-1] - period], c, [c[0] + period]])
    else:
        ext = np.concatenate([[2 * c[0] - c[1]], c, [2 * c[-1] - c[-2]]])
    e = 0.5 * (ext[:-1] + ext[1:])
    if clip is not None:
        e = np.clip(e, *clip)
    return e


def linear_matrix(src, tgt, periodic=False, period=360.0):
    """Row-stochastic linear-interpolation matrix, shape (len(tgt), len(src))."""
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    n, m = len(src), len(tgt)
    A = np.zeros((m, n))
    if periodic:
        s = np.concatenate([src, [src[0] + period]])
        for i, t in enumerate(tgt):
            tt = (t - src[0]) % period + src[0]
            k = int(np.clip(np.searchsorted(s, tt) - 1, 0, n - 1))
            w = (tt - s[k]) / (s[k + 1] - s[k])
            A[i, k] += 1 - w
            A[i, (k + 1) % n] += w
    else:
        for i, t in enumerate(tgt):
            k = int(np.clip(np.searchsorted(src, t) - 1, 0, n - 2))
            w = (t - src[k]) / (src[k + 1] - src[k])
            w = min(max(w, 0.0), 1.0)  # clamp -> nearest outside the source range
            A[i, k] += 1 - w
            A[i, k + 1] += w
    return A


def conservative_matrix(src, tgt, periodic=False, period=360.0, cos_weight=False):
    """Row-stochastic area-overlap (first-order conservative) matrix (len(tgt), len(src)).

    ``cos_weight`` uses d(sin phi) instead of d(phi) as the measure, i.e. true
    spherical area weighting for the latitude factor.
    """
    clip = (-90.0, 90.0) if cos_weight else None
    se = cell_edges(src, periodic, period, clip)
    te = cell_edges(tgt, periodic, period, clip)

    def measure(a, b):
        if cos_weight:
            return np.sin(np.deg2rad(b)) - np.sin(np.deg2rad(a))
        return b - a

    n, m = len(src), len(tgt)
    A = np.zeros((m, n))
    for i in range(m):
        t0, t1 = te[i], te[i + 1]
        for j in range(n):
            s0, s1 = se[j], se[j + 1]
            if periodic:
                best = 0.0
                for shift in (-period, 0.0, period):
                    lo, hi = max(t0, s0 + shift), min(t1, s1 + shift)
                    if hi > lo:
                        best += measure(lo, hi)
                ov = best
            else:
                lo, hi = max(t0, s0), min(t1, s1)
                ov = measure(lo, hi) if hi > lo else 0.0
            A[i, j] = ov
    rs = A.sum(1, keepdims=True)
    assert np.all(rs > 0), "empty target cell -- grids do not overlap"
    return A / rs


def cell_measures(centers, periodic=False, period=360.0, cos_weight=False):
    """Per-cell measure (area weight factor) for a 1-D coordinate."""
    clip = (-90.0, 90.0) if cos_weight else None
    e = cell_edges(centers, periodic, period, clip)
    if cos_weight:
        return np.sin(np.deg2rad(e[1:])) - np.sin(np.deg2rad(e[:-1]))
    return e[1:] - e[:-1]


# ---------------------------------------------------------------- operator


class SeparableCoarsener(torch.nn.Module):
    """A = A_lat (x) A_lon, with the exact consistency projection onto {A Y = X}.

    ``coarsen(Y)``  : (..., nlat_hr, nlon_hr) -> (..., nlat_lr, nlon_lr)
    ``upsample(X)`` : bilinear LR -> HR (the skip path, *not* a right inverse of A)
    ``project(Y, X)``: nearest point (in the area-weighted norm) to Y satisfying A Y = X

    The projection is
        Y* = Y + W^-1 A^T (A W^-1 A^T)^-1 (X - A Y),
    which for separable A and separable W is
        R    = X - A_lat Y A_lon^T
        S    = M_lat^-1 R M_lon^-1,     M = A D^-1 A^T
        Y*   = Y + D_lat^-1 A_lat^T S A_lon D_lon^-1
    and satisfies A Y* = X exactly (to floating-point round-off).
    """

    def __init__(self, mode="conservative", weighted=True, dtype=torch.float64,
                 lat_lr=None, lon_lr=None):
        super().__init__()
        self.mode = mode
        lat_lr = LAT_LR if lat_lr is None else np.asarray(lat_lr, dtype=np.float64)
        lon_lr = LON_LR if lon_lr is None else np.asarray(lon_lr, dtype=np.float64)
        self.lat_lr, self.lon_lr = lat_lr, lon_lr
        if mode == "conservative":
            A_lat = conservative_matrix(LAT_HR, lat_lr, cos_weight=True)
            A_lon = conservative_matrix(LON_HR, lon_lr, periodic=True)
        elif mode == "bilinear":
            A_lat = linear_matrix(LAT_HR, lat_lr)
            A_lon = linear_matrix(LON_HR, lon_lr, periodic=True)
        else:
            raise ValueError(mode)

        U_lat = linear_matrix(lat_lr, LAT_HR)
        U_lon = linear_matrix(lon_lr, LON_HR, periodic=True)

        if weighted:
            d_lat = cell_measures(LAT_HR, cos_weight=True)
            d_lon = cell_measures(LON_HR, periodic=True)
        else:
            d_lat = np.ones(len(LAT_HR))
            d_lon = np.ones(len(LON_HR))
        d_lat = d_lat / d_lat.mean()
        d_lon = d_lon / d_lon.mean()

        M_lat = A_lat @ np.diag(1.0 / d_lat) @ A_lat.T
        M_lon = A_lon @ np.diag(1.0 / d_lon) @ A_lon.T

        reg = lambda M: M + 1e-12 * np.trace(M) / len(M) * np.eye(len(M))
        for name, arr in [
            ("A_lat", A_lat), ("A_lon", A_lon),
            ("U_lat", U_lat), ("U_lon", U_lon),
            ("Mi_lat", np.linalg.inv(reg(M_lat))),
            ("Mi_lon", np.linalg.inv(reg(M_lon))),
            ("di_lat", 1.0 / d_lat), ("di_lon", 1.0 / d_lon),
        ]:
            self.register_buffer(name, torch.as_tensor(arr, dtype=dtype))

        self.cond = (np.linalg.cond(M_lat), np.linalg.cond(M_lon))

    def coarsen(self, Y):
        with exact_matmul():
            return torch.einsum("ij,...jk,lk->...il", self.A_lat, Y, self.A_lon)

    def upsample(self, X):
        with exact_matmul():
            return torch.einsum("ij,...jk,lk->...il", self.U_lat, X, self.U_lon)

    def project(self, Y, X):
        with exact_matmul():
            R = X - self.coarsen(Y)
            S = torch.einsum("ij,...jk,kl->...il", self.Mi_lat, R, self.Mi_lon)
            dY = torch.einsum("ji,...jk,kl->...il", self.A_lat, S, self.A_lon)
            dY = dY * self.di_lat[:, None] * self.di_lon[None, :]
            return Y + dY

    def consistency_error(self, Y, X):
        """RMS of A Y - X, per leading batch element."""
        r = self.coarsen(Y) - X
        return torch.sqrt((r ** 2).mean(dim=(-2, -1)))


def cellcentered_grid(deg):
    """Cell-centered global lat-lon grid at `deg` resolution, WeatherBench convention."""
    n_lat, n_lon = int(round(180 / deg)), int(round(360 / deg))
    return -90 + deg / 2 + deg * np.arange(n_lat), deg * np.arange(n_lon)


def area_weights(lat=LAT_HR):
    """Normalized cos(phi) latitude weights for lat-weighted metrics (mean 1)."""
    w = cell_measures(lat, cos_weight=True)
    return w / w.mean()
