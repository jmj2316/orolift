"""Do tropical cyclones survive 5.625 -> 1.5 downscaling?

At 5.625 deg a typhoon core spans well under one grid cell, so its pressure
minimum is almost entirely erased.  This scans the 2020 test year for the deepest
western-North-Pacific MSLP minima in the ERA5 1.5 deg truth and asks how much of
each one the model puts back.
"""
import os, sys, json, importlib.util
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, LAT_LR, LON_LR
from orolift.varlist import VARS_69
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

BOX = (105, 175, 5, 40)          # western North Pacific
dev = "cuda"


def main():
    mean, std = load_norm()
    mt = torch.tensor(mean, device=dev)[:, None, None]
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    model, _ = ev.load_model("runs/orolift_base/best.pt", np.load(f"{DATA}/resid_std.npy"), dev)
    ip, iu, iv = (VARS_69.index(v) for v in
                  ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"])

    m = (LAT_HR >= BOX[2]) & (LAT_HR <= BOX[3]); n = (LON_HR >= BOX[0]) & (LON_HR <= BOX[1])
    mL = (LAT_LR >= BOX[2]) & (LAT_LR <= BOX[3]); nL = (LON_LR >= BOX[0]) & (LON_LR <= BOX[1])
    ds = HR15("test")
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4)

    rec = []
    i0 = 0
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            p = model(x, sl, sh).float()
        b = C.project(C.upsample(x), x)
        P = lambda a: (a[:, ip] * st[ip] + mt[ip])[:, m][:, :, n]
        W = lambda a: torch.sqrt((a[:, iu] * st[iu] + mt[iu]) ** 2 +
                                 (a[:, iv] * st[iv] + mt[iv]) ** 2)[:, m][:, :, n]
        xp = (x[:, ip] * st[ip] + mt[ip])[:, mL][:, :, nL]
        for k in range(y.shape[0]):
            rec.append(dict(i=i0 + k, t=str(ds.times[i0 + k]),
                            p_true=float(P(y)[k].min()), p_base=float(P(b)[k].min()),
                            p_oro=float(P(p)[k].min()), p_lr=float(xp[k].min()),
                            w_true=float(W(y)[k].max()), w_base=float(W(b)[k].max()),
                            w_oro=float(W(p)[k].max())))
        i0 += y.shape[0]

    rec.sort(key=lambda r: r["p_true"])
    print(f"{'time':22s}{'truth':>9s}{'5.625° in':>11s}{'bil+proj':>10s}{'OroLIFT':>9s}"
          f"{'|  W truth':>11s}{'bil+proj':>10s}{'OroLIFT':>9s}")
    for r in rec[:12]:
        print(f"{r['t']:22s}{r['p_true']/100:9.1f}{r['p_lr']/100:11.1f}{r['p_base']/100:10.1f}"
              f"{r['p_oro']/100:9.1f}   {r['w_true']:8.1f}{r['w_base']:10.1f}{r['w_oro']:9.1f}")
    top = rec[:40]
    dt = np.array([r["p_true"] for r in top]); db = np.array([r["p_base"] for r in top])
    do = np.array([r["p_oro"] for r in top]); dl_ = np.array([r["p_lr"] for r in top])
    print(f"\n40 deepest WNP minima of 2020, mean central pressure (hPa):")
    print(f"   ERA5 1.5°  {dt.mean()/100:7.2f}   5.625° input {dl_.mean()/100:7.2f}   "
          f"bilinear+proj {db.mean()/100:7.2f}   OroLIFT {do.mean()/100:7.2f}")
    print(f"   mean |error| vs truth:  bilinear+proj {np.abs(db-dt).mean()/100:5.2f} hPa   "
          f"OroLIFT {np.abs(do-dt).mean()/100:5.2f} hPa")
    wt = np.array([r["w_true"] for r in top]); wb = np.array([r["w_base"] for r in top])
    wo = np.array([r["w_oro"] for r in top])
    print(f"   peak 10 m wind (m/s): truth {wt.mean():5.1f}   bil+proj {wb.mean():5.1f}   "
          f"OroLIFT {wo.mean():5.1f}")
    json.dump(rec, open("runs/typhoon.json", "w"))

    # ------------------------------------------------------------------ figure
    ti = rec[0]["i"]
    y = ds[ti][None].to(dev); x = C.coarsen(y)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        p = model(x, sl, sh).float()
    b = C.project(C.upsample(x), x)
    lo, la = np.meshgrid(LON_HR, LAT_HR); loL, laL = np.meshgrid(LON_LR, LAT_LR)
    lsm = sh[0, 3].cpu().numpy()
    gp = lambda a: (a[0, ip] * st[ip] + mt[ip]).cpu().numpy() / 100
    gw = lambda a: torch.sqrt((a[0, iu] * st[iu] + mt[iu]) ** 2 +
                              (a[0, iv] * st[iv] + mt[iv]) ** 2).cpu().numpy()
    # restrict every reported min/max to the basin box -- the global MSLP minimum
    # at this time sits in the Southern Ocean, not in the typhoon
    def boxstat(X, Y, F, how):
        msk = (Y >= BOX[2]) & (Y <= BOX[3]) & (X >= BOX[0]) & (X <= BOX[1])
        return float(F[msk].min() if how == "min" else F[msk].max())

    fields = [("5.625° input", loL, laL, (x[0, ip] * st[ip] + mt[ip]).cpu().numpy() / 100, None),
              ("bilinear + projection", lo, la, gp(b), gw(b)),
              ("OroLIFT", lo, la, gp(p), gw(p)),
              ("ERA5 1.5° (truth)", lo, la, gp(y), gw(y))]
    fig, ax = plt.subplots(2, 4, figsize=(12, 5.0), constrained_layout=True)
    vmin = boxstat(lo, la, gp(y), "min") - 2; vmax = 1016
    wmax = boxstat(lo, la, gw(y), "max")
    for j, (t, X, Y, Pf, Wf) in enumerate(fields):
        im = ax[0, j].pcolormesh(X, Y, Pf, cmap="viridis", vmin=vmin, vmax=vmax,
                                 shading="nearest" if j == 0 else "auto")
        ax[0, j].contour(X, Y, Pf, levels=np.arange(955, 1020, 4), colors="w", linewidths=0.4)
        ax[0, j].contour(lo, la, lsm, levels=[0.5], colors="k", linewidths=0.6)
        ax[0, j].set_xlim(BOX[:2]); ax[0, j].set_ylim(BOX[2:])
        ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
        ax[0, j].set_title(f"{t}\nbasin min = {boxstat(X, Y, Pf, 'min'):.1f} hPa")
        if Wf is not None:
            imw = ax[1, j].pcolormesh(X, Y, Wf, cmap="magma_r", vmin=0, vmax=wmax, shading="auto")
            ax[1, j].contour(lo, la, lsm, levels=[0.5], colors="w", linewidths=0.6)
            ax[1, j].set_title(f"basin max 10 m wind = {boxstat(X, Y, Wf, 'max'):.1f} m/s")
        else:
            wx = torch.sqrt((x[0, iu] * st[iu] + mt[iu]) ** 2 +
                            (x[0, iv] * st[iv] + mt[iv]) ** 2).cpu().numpy()
            imw = ax[1, j].pcolormesh(loL, laL, wx, cmap="magma_r", vmin=0, vmax=wmax,
                                      shading="nearest")
            ax[1, j].contour(lo, la, lsm, levels=[0.5], colors="w", linewidths=0.6)
            ax[1, j].set_title(f"basin max 10 m wind = {boxstat(loL, laL, wx, 'max'):.1f} m/s")
        ax[1, j].set_xlim(BOX[:2]); ax[1, j].set_ylim(BOX[2:])
        ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
    fig.colorbar(im, ax=ax[0, :], shrink=0.85, label="MSLP (hPa)", pad=0.01)
    fig.colorbar(imw, ax=ax[1, :], shrink=0.85, label="10 m wind (m/s)", pad=0.01)
    fig.suptitle(f"Deepest western North Pacific minimum of 2020 — {rec[0]['t']}", y=1.04)
    fig.savefig("figures/fig7_typhoon.png", bbox_inches="tight")
    print("\nwrote figures/fig7_typhoon.png")


if __name__ == "__main__":
    main()
