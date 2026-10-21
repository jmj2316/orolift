"""Gallery of 2020 tropical cyclone cases, western North Pacific + North Atlantic.

Events are found by scanning the test year for basin MSLP minima and collapsing
timesteps within 2 days of each other into one storm, so the panels are distinct
cyclones rather than six snapshots of the same one.
"""
import os, sys, importlib.util
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, LAT_LR, LON_LR
from orolift.varlist import VARS_69, LEVELS
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
BASINS = {"W. North Pacific": (105, 175, 5, 30), "North Atlantic": (260, 350, 5, 30)}
SEASON = range(6, 12)          # Jun-Nov: both basins' TC season
HALF = 15.0          # degrees of zoom around each center


def main():
    mean, std = load_norm()
    mt = torch.tensor(mean, device=dev)[:, None, None]
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    model, _ = ev.load_model("runs/orolift_base/best.pt", np.load(f"{DATA}/resid_std.npy"), dev)
    ip = VARS_69.index("mean_sea_level_pressure")
    it3 = VARS_69.index("temperature_300")   # warm-core test: TCs are warm aloft,
    it5 = VARS_69.index("temperature_500")   # extratropical lows are cold-core
    lsm = sh[0, 3].cpu().numpy()
    ds = HR15("test")
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4)

    recs = []
    i0 = 0
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            p = model(x, sl, sh).float()
        b = C.project(C.upsample(x), x)
        f = lambda a: (a[:, ip] * st[ip] + mt[ip]).cpu().numpy() / 100
        Py, Pb, Pp = f(y), f(b), f(p)
        Tm = ((y[:, it3] * st[it3] + mt[it3]) + (y[:, it5] * st[it5] + mt[it5])).cpu().numpy() / 2
        Px = (x[:, ip] * st[ip] + mt[ip]).cpu().numpy() / 100
        for bn, (l0, l1, a0, a1) in BASINS.items():
            m = ((LAT_HR >= a0) & (LAT_HR <= a1))[:, None] & ((LON_HR >= l0) & (LON_HR <= l1))[None, :]
            mL = ((LAT_LR >= a0) & (LAT_LR <= a1))[:, None] & ((LON_LR >= l0) & (LON_LR <= l1))[None, :]
            for k in range(y.shape[0]):
                if int(str(ds.times[i0 + k])[5:7]) not in SEASON:
                    continue
                jj, ii = np.unravel_index(np.argmin(np.where(m, Py[k], 1e9)), Py[k].shape)
                # warm core: mid-tropospheric T within ~330 km of the center against
                # the 5-15 deg annulus, on the same latitude circle
                d2 = ((LAT_HR[:, None] - LAT_HR[jj]) ** 2 +
                      ((LON_HR[None, :] - LON_HR[ii]) * np.cos(np.deg2rad(LAT_HR[jj]))) ** 2)
                core = Tm[k][d2 < 3.0 ** 2].mean()
                env = Tm[k][(d2 > 5.0 ** 2) & (d2 < 15.0 ** 2)].mean()
                if core - env < 0.5:
                    continue
                recs.append(dict(basin=bn, i=i0 + k, t=str(ds.times[i0 + k]), warm=float(core - env),
                                 lat=float(LAT_HR[jj]), lon=float(LON_HR[ii]),
                                 true=float(Py[k][jj, ii]),
                                 base=float(np.where(m, Pb[k], 1e9).min()),
                                 oro=float(np.where(m, Pp[k], 1e9).min()),
                                 lr=float(np.where(mL, Px[k], 1e9).min())))
        i0 += y.shape[0]

    # ---- collapse to distinct storms (>2 days apart within a basin)
    recs.sort(key=lambda r: r["true"])
    def same_storm(a, c):
        """Same cyclone if close in both time and space (6-hourly index, degrees)."""
        d = np.hypot(a["lat"] - c["lat"], (a["lon"] - c["lon"]) * np.cos(np.deg2rad(a["lat"])))
        return a["basin"] == c["basin"] and abs(a["i"] - c["i"]) < 28 and d < 30

    picked = []
    for r in recs:
        if all(not same_storm(q, r) for q in picked):
            picked.append(r)
        if len(picked) == 6:
            break
    for r in picked:
        print(f"{r['basin']:18s} {r['t']:20s} {r['lat']:5.1f}N {r['lon']:6.1f}E  warm+{r['warm']:.1f}K   "
              f"truth {r['true']:7.1f}  input {r['lr']:7.1f}  base {r['base']:7.1f}  OroLIFT {r['oro']:7.1f}")

    # ---- gallery
    fig, ax = plt.subplots(6, 4, figsize=(9.6, 12.6), gridspec_kw=dict(wspace=0.04, hspace=0.07))
    lo, la = np.meshgrid(LON_HR, LAT_HR); loL, laL = np.meshgrid(LON_LR, LAT_LR)
    HEAD = ["5.625° input", "Bilinear + proj.", "OroLIFT", "ERA5 1.5° (truth)"]
    for row, r in enumerate(picked):
        y = ds[r["i"]][None].to(dev); x = C.coarsen(y)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            p = model(x, sl, sh).float()
        b = C.project(C.upsample(x), x)
        g = lambda a: (a[0, ip] * st[ip] + mt[ip]).cpu().numpy() / 100
        box = (r["lon"] - HALF, r["lon"] + HALF, r["lat"] - HALF, r["lat"] + HALF)
        vmin, vmax = r["true"] - 2, 1014
        panels = [(loL, laL, (x[0, ip] * st[ip] + mt[ip]).cpu().numpy() / 100, r["lr"]),
                  (lo, la, g(b), r["base"]), (lo, la, g(p), r["oro"]), (lo, la, g(y), r["true"])]
        for col, (X, Y, F, mn) in enumerate(panels):
            a = ax[row, col]
            im = a.pcolormesh(X, Y, F, cmap="viridis", vmin=vmin, vmax=vmax,
                              shading="nearest" if col == 0 else "auto", rasterized=True)
            a.contour(X, Y, F, levels=np.arange(round(vmin / 4) * 4, 1020, 4), colors="w", linewidths=0.35)
            a.contour(lo, la, lsm, levels=[0.5], colors="k", linewidths=0.6)
            a.set_xlim(box[:2]); a.set_ylim(box[2:]); a.set_xticks([]); a.set_yticks([])
            if row == 0:
                a.set_title(HEAD[col], fontsize=10, color=(S.ACCENT if col == 2 else "k"),
                            fontweight=("bold" if col == 2 else "normal"))
            a.text(0.04, 0.04, f"{mn:.0f} hPa", transform=a.transAxes, fontsize=8.5, ha="left", va="bottom",
                   color=(S.ACCENT if col == 2 else "k"), fontweight=("bold" if col == 2 else "normal"),
                   bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.88), zorder=10)
            if col == 2:
                for sp in a.spines.values():
                    sp.set_edgecolor(S.ACCENT); sp.set_linewidth(1.6)
            if col == 0:
                a.set_ylabel(f"{r['t'][:10]} {r['t'][11:13]}Z\n{r['lat']:.1f}°N {r['lon']:.1f}°E\n{r['basin']}", fontsize=8)
                a.text(0.04, 0.95, str(row + 1), transform=a.transAxes, fontsize=11, fontweight="bold", ha="left", va="top", bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.88), zorder=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.015, aspect=30)
    cb.set_label("mean sea-level pressure (hPa)")
    fig.savefig("figures/fig9_typhoon_gallery.png")
    plt.close(fig)
    print("wrote figures/fig9_typhoon_gallery.png")

    # ---- scatter over all distinct events
    ev_all = []
    for r in recs:
        if all(not same_storm(q, r) for q in ev_all):
            ev_all.append(r)
    ev_all = [r for r in ev_all if r["true"] < 1000]
    t_ = np.array([r["true"] for r in ev_all]); b_ = np.array([r["base"] for r in ev_all])
    o_ = np.array([r["oro"] for r in ev_all]); l_ = np.array([r["lr"] for r in ev_all])
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9), gridspec_kw=dict(wspace=0.27))
    lim = (t_.min() - 3, 1002)
    ax[0].plot(lim, lim, "-", color="0.2", lw=0.9, zorder=1)
    ax[0].scatter(t_, l_, s=22, color=S.PAL["x"], marker="^", alpha=0.9, label=f"5.625° input (MAE {np.abs(l_-t_).mean():.1f} hPa)")
    ax[0].scatter(t_, b_, s=22, color=S.PAL["bilinear+proj"], marker="s", alpha=0.95, label=f"Bilinear + proj. (MAE {np.abs(b_-t_).mean():.1f} hPa)")
    ax[0].scatter(t_, o_, s=26, color=S.ACCENT, alpha=0.95, label=f"OroLIFT (MAE {np.abs(o_-t_).mean():.1f} hPa)")
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("ERA5 1.5° central pressure (hPa)"); ax[0].set_ylabel("reconstructed central pressure (hPa)")
    ax[0].legend(loc="lower right", fontsize=8.2, handletextpad=0.3)
    ax[0].set_title(f"{len(ev_all)} distinct 2020 cyclones", fontsize=9.5)
    ax[1].axhline(0, color="0.2", lw=0.9, zorder=1)
    ax[1].scatter(t_, b_ - t_, s=22, color=S.PAL["bilinear+proj"], marker="s", alpha=0.95, label="Bilinear + proj.")
    ax[1].scatter(t_, o_ - t_, s=26, color=S.ACCENT, alpha=0.95, label="OroLIFT")
    ax[1].set_xlabel("ERA5 1.5° central pressure (hPa)"); ax[1].set_ylabel("error (hPa)")
    ax[1].legend(loc="upper right", fontsize=8.2, handletextpad=0.3)
    ax[1].set_title("Bias against storm depth", fontsize=9.5)
    for a, t in zip(ax, "ab"):
        a.spines[["top", "right"]].set_visible(False); S.tag(a, t, dx=0.0, dy=1.08)
    fig.savefig("figures/fig10_typhoon_scatter.png", bbox_inches="tight"); plt.close(fig)
    print("wrote figures/fig10_typhoon_scatter.png")


if __name__ == "__main__":
    main()
