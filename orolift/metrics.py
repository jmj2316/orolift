"""Evaluation metrics for OroLIFT.

Everything here takes *physical-unit* fields of shape (B, V, 121, 240) unless
stated otherwise, and returns per-variable tensors of shape (V,).
"""
import numpy as np
import torch

from .grids import LAT_HR, LON_HR, area_weights
from .varlist import VARS_69, LEVELS

R_DRY = 287.05
OMEGA = 7.292115e-5
R_EARTH = 6371.0e3


def _w(device, dtype):
    return torch.as_tensor(area_weights(), device=device, dtype=dtype)[:, None]


def lat_weighted_rmse(pred, true):
    w = _w(pred.device, pred.dtype)
    return torch.sqrt((w * (pred - true) ** 2).mean(dim=(-2, -1))).mean(0)


def lat_weighted_bias(pred, true):
    w = _w(pred.device, pred.dtype)
    return (w * (pred - true)).mean(dim=(-2, -1)).mean(0)


def lat_weighted_acc(pred, true, clim):
    """Anomaly correlation against a per-gridpoint climatology (B,V,H,W) or (V,H,W)."""
    w = _w(pred.device, pred.dtype)
    pa, ta = pred - clim, true - clim
    num = (w * pa * ta).sum(dim=(-2, -1))
    den = torch.sqrt((w * pa ** 2).sum(dim=(-2, -1)) * (w * ta ** 2).sum(dim=(-2, -1)))
    return (num / den.clamp_min(1e-30)).mean(0)


# ------------------------------------------------------------------ spectra

def zonal_spectrum(x, lat_band=(-60.0, 60.0)):
    """Mean zonal power spectrum over a latitude band.  (B,V,H,W) -> (V, W//2+1)."""
    m = (LAT_HR >= lat_band[0]) & (LAT_HR <= lat_band[1])
    sel = x[..., m, :]
    f = torch.fft.rfft(sel.double(), dim=-1)
    p = (f.abs() ** 2) / sel.shape[-1]
    return p.mean(dim=(0, -2))


def spectral_ratio(pred, true, kmin=32, lat_band=(-60.0, 60.0)):
    """Mean power ratio pred/true over wavenumbers >= kmin.

    1.0 is perfect; a deterministic regressor sits well below 1 (over-smoothing).
    kmin=32 is the Nyquist wavenumber of the 5.625-degree input grid, i.e. exactly
    the band that downscaling has to invent.
    """
    ps, pt = zonal_spectrum(pred, lat_band), zonal_spectrum(true, lat_band)
    return (ps[:, kmin:].sum(-1) / pt[:, kmin:].sum(-1).clamp_min(1e-30))


# ------------------------------------------------------------------ physics

_LEV = {f"{p}_{l}": (p, l) for p in
        ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"]
        for l in LEVELS}


def _idx(prefix):
    return [VARS_69.index(f"{prefix}_{l}") for l in LEVELS]


def hydrostatic_residual(x):
    """RMS violation of dPhi/dlnp = -R T between adjacent pressure levels.

    Returns (B, n_levels-1) in m^2/s^2.  Compared model-vs-truth, not to zero:
    ERA5 itself has a small residual from the discrete level spacing.
    """
    zi, ti = _idx("geopotential"), _idx("temperature")
    p = torch.tensor(LEVELS, device=x.device, dtype=x.dtype)
    Z, T = x[:, zi], x[:, ti]
    dlnp = torch.log(p[1:] / p[:-1])                      # >0, going downward
    Tbar = 0.5 * (T[:, 1:] + T[:, :-1])
    lhs = Z[:, :-1] - Z[:, 1:]                            # upper minus lower  (>0)
    rhs = R_DRY * Tbar * dlnp[None, :, None, None]
    w = _w(x.device, x.dtype)
    return torch.sqrt((w * (lhs - rhs) ** 2).mean(dim=(-2, -1)))


def geostrophic_residual(x, lat_band=(25.0, 70.0)):
    """RMS ageostrophic wind |v - v_g| in the extratropics, per level (B, n_levels).

    v_g is diagnosed from the model's own geopotential, so this asks whether the
    downscaled Z and (u, v) fields remain mutually balanced.
    """
    zi, ui, vi = _idx("geopotential"), _idx("u_component_of_wind"), _idx("v_component_of_wind")
    Z, U, V = x[:, zi], x[:, ui], x[:, vi]
    lat = torch.as_tensor(LAT_HR, device=x.device, dtype=x.dtype)
    f = 2 * OMEGA * torch.sin(torch.deg2rad(lat))
    dlat = torch.deg2rad(torch.as_tensor(LAT_HR[1] - LAT_HR[0], device=x.device, dtype=x.dtype))
    dlon = torch.deg2rad(torch.as_tensor(LON_HR[1] - LON_HR[0], device=x.device, dtype=x.dtype))
    coslat = torch.cos(torch.deg2rad(lat)).clamp_min(1e-6)

    dZdy = torch.gradient(Z, dim=-2)[0] / (R_EARTH * dlat)
    dZdx = (torch.roll(Z, -1, -1) - torch.roll(Z, 1, -1)) / (2 * R_EARTH * dlon * coslat[:, None])
    ug = -dZdy / f[:, None]
    vg = dZdx / f[:, None]

    m = torch.as_tensor((np.abs(LAT_HR) >= lat_band[0]) & (np.abs(LAT_HR) <= lat_band[1]),
                        device=x.device)
    ag = ((U - ug) ** 2 + (V - vg) ** 2)[..., m, :]
    return torch.sqrt(ag.mean(dim=(-2, -1)))


# ------------------------------------------------------------------ extremes / terrain

def quantile_error(pred, true, q=0.99):
    """Error of the per-gridpoint q-quantile over the batch, lat-weighted RMS."""
    w = _w(pred.device, pred.dtype)
    qp = torch.quantile(pred.float(), q, dim=0)
    qt = torch.quantile(true.float(), q, dim=0)
    return torch.sqrt((w * (qp - qt) ** 2).mean(dim=(-2, -1)))


def terrain_stratified_rmse(pred, true, oro_detail, edges=(0.0, 100.0, 400.0, 1e9)):
    """RMSE split by |sub-grid orography| -- the bins where downscaling should pay off.

    ``oro_detail`` is the HR orography minus its coarsened-then-upsampled version,
    in meters.  Returns (n_bins, V).
    """
    a = oro_detail.abs()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a >= lo) & (a < hi)
        if m.sum() == 0:
            out.append(torch.full((pred.shape[1],), float("nan"), device=pred.device))
            continue
        d = ((pred - true) ** 2)[..., m]
        out.append(torch.sqrt(d.mean(dim=(0, -1))))
    return torch.stack(out)
