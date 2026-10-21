"""Dataset / static features for OroLIFT.

The store holds z-scored HR fields only; the LR input is produced on the fly by
the separable coarsener, which is both cheaper than storing it and the only way
the consistency certificate can be exact.  Normalization is a per-variable affine
map with constant coefficients, so it commutes with the (row-stochastic) coarsener
and with the projection -- everything can stay in normalized space.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .grids import SeparableCoarsener, LAT_HR, LON_HR
from .varlist import VARS_69

DATA = os.environ.get("OROLIFT_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
# the large float16 stores can live on another disk: set OROLIFT_BIG
BIG = os.environ.get("OROLIFT_BIG", DATA)
NV, NLAT, NLON = 69, 121, 240


def load_norm():
    d = np.load(f"{DATA}/norm.npz", allow_pickle=True)
    assert list(d["vars"]) == VARS_69
    return d["mean"].astype(np.float32), d["std"].astype(np.float32)


class HR15(Dataset):
    def __init__(self, split, root=None, preload=False, limit=None):
        root = root or BIG
        path = f"{root}/hr15_{split}.f16"
        n = os.path.getsize(path) // (NV * NLAT * NLON * 2)
        self.n = n if limit is None else min(n, limit)
        self.path, self.split = path, split
        tp = f"{root}/times_{split}.npy"
        if not os.path.exists(tp):          # big memmaps may live on a node-local disk
            tp = f"{DATA}/times_{split}.npy"
        self.times = np.load(tp)[: self.n]
        self._mm = None
        if preload:                       # NFS: read once, sequentially, then share
            self._mm = np.fromfile(path, np.float16,
                                   count=self.n * NV * NLAT * NLON
                                   ).reshape(self.n, NV, NLAT, NLON)

    def _arr(self):
        if self._mm is None:
            self._mm = np.memmap(self.path, np.float16, "r",
                                 shape=(self.n, NV, NLAT, NLON))
        return self._mm

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self._arr()[i], np.float32))


def static_features(device="cpu", dtype=torch.float32, mode="conservative",
                    lat_lr=None, lon_lr=None):
    """HR and LR static predictors.

    HR (7 ch): orography (km), sub-grid orography detail (km), |grad oro|,
               land-sea mask, sin(lat), cos(lat)*cos(lon), cos(lat)*sin(lon)
    LR (4 ch): orography, land-sea mask, sin(lat), cos(lat) -- plus the two
               longitude channels, for symmetry with the HR set.

    The *detail* channel is the whole point: it is exactly the topographic
    information that the 5.625-degree input cannot see.
    """
    d = np.load(f"{DATA}/static_hr.npz")
    oro = torch.as_tensor(d["geopotential_at_surface"] / 9.80665 / 1000.0, dtype=dtype)  # km
    lsm = torch.as_tensor(d["land_sea_mask"], dtype=dtype)

    C = SeparableCoarsener(mode, dtype=torch.float64, lat_lr=lat_lr, lon_lr=lon_lr)
    oro_lr = C.coarsen(oro.double())
    lsm_lr = C.coarsen(lsm.double())
    oro_det = (oro.double() - C.upsample(oro_lr)).to(dtype)

    gy = torch.gradient(oro, dim=0)[0]
    gx = torch.gradient(oro, dim=1)[0]
    grad = torch.sqrt(gy ** 2 + gx ** 2)

    lat = torch.as_tensor(np.deg2rad(LAT_HR), dtype=dtype)[:, None].expand(NLAT, NLON)
    lon = torch.as_tensor(np.deg2rad(LON_HR), dtype=dtype)[None, :].expand(NLAT, NLON)
    hr = torch.stack([oro, oro_det, grad * 10.0, lsm,
                      torch.sin(lat), torch.cos(lat) * torch.cos(lon),
                      torch.cos(lat) * torch.sin(lon)])

    nla, nlo = len(C.lat_lr), len(C.lon_lr)
    latL = torch.as_tensor(np.deg2rad(C.lat_lr), dtype=dtype)[:, None].expand(nla, nlo)
    lonL = torch.as_tensor(np.deg2rad(C.lon_lr), dtype=dtype)[None, :].expand(nla, nlo)
    lr = torch.stack([oro_lr.to(dtype), lsm_lr.to(dtype),
                      torch.sin(latL), torch.cos(latL) * torch.cos(lonL),
                      torch.cos(latL) * torch.sin(lonL)])
    return hr.to(device), lr.to(device)


def hr_coords(dtype=torch.float32):
    """HR query coordinates in [-1, 1]^2 (lat, lon) for the implicit decoder."""
    la = torch.as_tensor(LAT_HR / 90.0, dtype=dtype)[:, None].expand(NLAT, NLON)
    lo = torch.as_tensor(LON_HR / 180.0 - 1.0, dtype=dtype)[None, :].expand(NLAT, NLON)
    return torch.stack([la, lo])
