"""Reference numbers: what the LR input alone already gives you, and what the
consistency projection does to it.  Everything downstream is measured against
this table.
"""
import argparse, os, sys, json
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features
from orolift.grids import SeparableCoarsener
from orolift.metrics import (lat_weighted_rmse, spectral_ratio, hydrostatic_residual,
                             geostrophic_residual, terrain_stratified_rmse)
from orolift.varlist import VARS_69

KEY = ["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure",
       "geopotential_500", "temperature_850", "u_component_of_wind_850",
       "specific_humidity_700", "temperature_1000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="conservative")
    ap.add_argument("--out", default="runs/baseline.json")
    a = ap.parse_args()

    dev = a.device
    mean, std = load_norm()
    mean_t = torch.tensor(mean, device=dev)[:, None, None]
    std_t = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener(a.mode, dtype=torch.float32).to(dev)
    hr_static, _ = static_features(dev)
    oro_det_m = hr_static[1] * 1000.0                      # km -> m

    ds = HR15(a.split, limit=a.n)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=6, shuffle=False)

    METHODS = ["bilinear", "bilinear+proj"]
    acc = {m: {k: [] for k in ["rmse", "spec", "hyd", "geo", "cons"]} for m in METHODS}
    acc["truth"] = {"hyd": [], "geo": []}
    terr = {m: [] for m in METHODS}
    nb = 0
    for yb in dl:
        y = yb.to(dev, non_blocking=True)                   # normalized
        x = C.coarsen(y)                                    # LR input
        preds = {"bilinear": C.upsample(x)}
        preds["bilinear+proj"] = C.project(preds["bilinear"], x)
        yp = y * std_t + mean_t
        for m in METHODS:
            pp = preds[m] * std_t + mean_t
            acc[m]["rmse"].append(lat_weighted_rmse(pp, yp).cpu())
            acc[m]["spec"].append(spectral_ratio(pp, yp).cpu())
            acc[m]["hyd"].append(hydrostatic_residual(pp).mean(0).cpu())
            acc[m]["geo"].append(geostrophic_residual(pp).mean(0).cpu())
            acc[m]["cons"].append(C.consistency_error(preds[m], x).mean(0).cpu())
            terr[m].append(terrain_stratified_rmse(pp, yp, oro_det_m).cpu())
        acc["truth"]["hyd"].append(hydrostatic_residual(yp).mean(0).cpu())
        acc["truth"]["geo"].append(geostrophic_residual(yp).mean(0).cpu())
        nb += 1
    red = {m: {k: torch.stack(v).mean(0) for k, v in d.items()} for m, d in acc.items()}
    terr = {m: torch.stack(v).mean(0) for m, v in terr.items()}

    print(f"\n=== {a.split}: {len(ds)} samples, operator={a.mode} ===")
    print(f"{'variable':26s} " + "".join(f"{m:>16s}" for m in ["bilinear", "bilinear+proj"])
          + f"{'spec.ratio(bil)':>16s}")
    for v in KEY:
        i = VARS_69.index(v)
        print(f"{v:26s} " + "".join(f"{red[m]['rmse'][i]:16.4g}" for m in ["bilinear", "bilinear+proj"])
              + f"{red['bilinear']['spec'][i]:16.3f}")
    print("\nconsistency ||A y - x|| (normalized units):")
    for m in ["bilinear", "bilinear+proj"]:
        print(f"   {m:16s} {red[m]['cons'].mean():.3e}")
    print("\nphysics residuals (mean over levels):")
    print(f"   hydrostatic  truth={red['truth']['hyd'].mean():10.4g}  "
          f"bilinear={red['bilinear']['hyd'].mean():10.4g}  proj={red['bilinear+proj']['hyd'].mean():10.4g}")
    print(f"   ageostrophic truth={red['truth']['geo'].mean():10.4g}  "
          f"bilinear={red['bilinear']['geo'].mean():10.4g}  proj={red['bilinear+proj']['geo'].mean():10.4g}")
    print("\nRMSE by sub-grid orography |detail| (m): [0-100) [100-400) [400+)")
    for v in ["2m_temperature", "temperature_1000", "u_component_of_wind_850"]:
        i = VARS_69.index(v)
        print(f"   {v:24s} " + "".join(f"{terr['bilinear'][b][i]:10.4g}" for b in range(3)))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({m: {k: v.tolist() for k, v in d.items()} for m, d in red.items()},
              open(a.out, "w"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
