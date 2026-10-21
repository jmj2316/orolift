"""Per-pixel error statistics and case fields for the five-method comparison figures.

Methods use the 25k-step comparison budget of Table 1 (cmp_*), so the maps agree with it.
Outputs figures/fig_stats.npz
"""
import os, sys, datetime as dt
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR
from orolift.metrics import terrain_stratified_rmse
from orolift.varlist import VARS_69
import importlib.util
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

DEV = "cuda"
METHODS = ["bilinear", "bilinear+proj", "liif", "edsr", "orolift"]
CK = {"liif": "runs/cmp_liif/best.pt", "edsr": "runs/cmp_edsr/best.pt", "orolift": "runs/cmp_orolift/best.pt"}
STAT_VARS = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
             "10m_v_component_of_wind", "temperature_850", "specific_humidity_700", "geopotential_500"]
CASE_VARS = ["2m_temperature", "mean_sea_level_pressure", "specific_humidity_700",
             "10m_u_component_of_wind", "10m_v_component_of_wind"]
CASES = ["2020-01-16 00:00:00", "2020-04-16 06:00:00", "2020-07-16 12:00:00",
         "2020-09-05 12:00:00", "2020-10-16 18:00:00"]

mean, std = load_norm()
mt = torch.tensor(mean, device=DEV)[:, None, None]; st = torch.tensor(std, device=DEV)[:, None, None]
C = SeparableCoarsener("conservative", dtype=torch.float32).to(DEV)
sh, sl = static_features(DEV); sh, sl = sh[None], sl[None]
rs = np.load(f"{DATA}/resid_std.npy")
models = {k: ev.load_model(v, rs, DEV)[0] for k, v in CK.items()}
ds = HR15("test")
times = [str(t) for t in ds.times]
cd = np.load(f"{DATA}/clim_doy.npy", mmap_mode="r")
doy = np.array([dt.date(*map(int, t[:10].split("-"))).timetuple().tm_yday - 1 for t in times])
iS = [VARS_69.index(v) for v in STAT_VARS]
iC = [VARS_69.index(v) for v in CASE_VARS]
w = torch.tensor(np.cos(np.deg2rad(LAT_HR)), device=DEV, dtype=torch.float64); w = w / w.mean()

NV, H, W = len(iS), 121, 240
sse = {m: torch.zeros(NV, H, W, dtype=torch.float64, device=DEV) for m in METHODS}
err = {m: torch.zeros(NV, H, W, dtype=torch.float64, device=DEV) for m in METHODS}
sst = torch.zeros(NV, H, W, dtype=torch.float64, device=DEV)
ter = {m: [] for m in METHODS}
case_idx = {times.index(c): c for c in CASES}
cases = {}
dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4)
i0 = 0
for y in dl:
    y = y.to(DEV); x = C.coarsen(y)
    up = C.upsample(x)
    out = {"bilinear": up, "bilinear+proj": C.project(up, x)}
    for k, m in models.items():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out[k] = m(x, sl, sh).float()
    yy = y * st + mt
    cb = torch.as_tensor(np.asarray(cd[doy[i0:i0 + y.shape[0]]], np.float32), device=DEV) * st + mt
    sst += ((yy - cb)[:, iS] ** 2).double().sum(0)
    for k in METHODS:
        pp = out[k] * st + mt
        d = (pp - yy)[:, iS].double()
        sse[k] += (d ** 2).sum(0); err[k] += d.sum(0)
        ter[k].append(terrain_stratified_rmse(pp, yy, sh[0, 1] * 1000.0).cpu())
    for j in range(y.shape[0]):
        if i0 + j in case_idx:
            cases[case_idx[i0 + j]] = dict(
                truth=yy[j, iC].cpu().numpy(), x=(x[j] * st[0:1] * 0 + x[j] * st + mt)[iC].cpu().numpy(),
                **{k: (out[k][j] * st + mt)[iC].cpu().numpy() for k in METHODS})
    i0 += y.shape[0]
    if (i0 // 16) % 20 == 0:
        print(i0, flush=True)

N = len(ds)
res = dict(n=N, stat_vars=np.array(STAT_VARS), methods=np.array(METHODS),
           sst=(sst / N).cpu().numpy(),
           sse=np.stack([(sse[m] / N).cpu().numpy() for m in METHODS]),
           mbe=np.stack([(err[m] / N).cpu().numpy() for m in METHODS]),
           terrain=np.stack([torch.stack(ter[m]).mean(0).numpy() for m in METHODS]),
           case_vars=np.array(CASE_VARS), case_times=np.array(list(cases.keys())))
for t, d in cases.items():
    for k, v in d.items():
        res[f"case|{t}|{k}"] = v.astype(np.float32)
np.savez("figures/fig_stats.npz", **res)
print("done", {k: np.shape(v) for k, v in res.items() if not k.startswith("case|")})
