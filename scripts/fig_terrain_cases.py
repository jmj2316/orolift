"""Terrain-class map + grouped bars, and case studies with zoom boxes (from figures/fig_stats.npz)."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from orolift.data import static_features
from orolift.varlist import VARS_69

D = np.load("figures/fig_stats.npz", allow_pickle=True)
M = list(D["methods"])
LAT = np.linspace(-90, 90, 121); LON = np.arange(240) * 1.5
hr, _ = static_features("cpu")
DET = hr[1].numpy() * 1000.0; LSM = hr[3].numpy()
LAT_LR = -87.1875 + 5.625 * np.arange(32); LON_LR = 5.625 * np.arange(64)
SH = 120
LONR = np.where(np.roll(LON, -SH) >= 180, np.roll(LON, -SH) - 360, np.roll(LON, -SH))
roll = lambda a: np.roll(a, -SH, axis=-1)
EXT = [LONR.min() - .75, LONR.max() + .75, -90.75, 90.75]


def fig_in(fig, x, y, w, h):
    W_, H_ = fig.get_size_inches()
    return fig.add_axes([x / W_, 1 - (y + h) / H_, w / W_, h / H_])


def terrain():
    a = np.abs(DET)
    cls = np.where(a < 100, 0, np.where(a < 400, 1, 2)).astype(float)
    frac = [float((np.cos(np.deg2rad(LAT))[:, None] * ((cls == k) & (LSM > 0.5))).sum() /
                  (np.cos(np.deg2rad(LAT))[:, None] * (LSM > 0.5)).sum() * 100) for k in range(3)]
    cols = ["#DCE6F5", "#6F94CF", "#1B2A5E"]
    cmap = ListedColormap(cols)
    fig = plt.figure(figsize=(10.4, 6.3))
    ax = fig_in(fig, 0.3, 0.45, 4.15, 2.1)
    ax.imshow(np.where(LSM > 0.5, roll(cls), np.nan) if False else roll(np.where(LSM > 0.5, cls, np.nan)), origin="lower",
              extent=EXT, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.contour(LONR, LAT, roll(LSM), levels=[0.5], colors="k", linewidths=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("terrain class (|sub-grid orography|)", fontsize=9.5, pad=3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=c, ec="0.4", lw=0.4, label=l + f"  ({f:.0f}% of land)") for c, l, f in
                       zip(cols, ["< 100 m", "100–400 m", "> 400 m"], frac)],
              loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="0.6",
              handlelength=1.1, borderpad=0.4, labelspacing=0.3)
    fig.text(0.012, 1 - 0.43 / 6.3, "a", fontsize=13, fontweight="bold", va="bottom")

    T = D["terrain"]
    names = ["< 100", "100–400", "> 400"]

    def bars(ax, var, ylabel, scale=1.0, ptag=None, lab=True, title=None):
        i = VARS_69.index(var)
        x = np.arange(3); w = 0.17
        for m in range(5):
            v = T[m, :, i] * scale
            b = ax.bar(x + (m - 2) * w, v, w * 0.92, color=S.PAL[M[m]], edgecolor="white", linewidth=0.3,
                       label=S.LABEL[M[m]] if M[m] != "edsr" else "EDSR", zorder=3)
            if M[m] in ("edsr", "orolift"):
                for xi, vi in zip(x + (m - 2) * w, v):
                    ax.text(xi, vi * 1.02, f"{vi:.3g}", rotation=90, ha="center", va="bottom", fontsize=8,
                            color=S.PAL[M[m]] if M[m] == "orolift" else "0.2", fontweight="bold" if M[m] == "orolift" else "normal")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_xlabel("|sub-grid orography| (m)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if title: ax.set_title(title, fontsize=9.5, pad=3)

    axb = fig_in(fig, 5.45, 0.55, 4.85, 1.95)
    bars(axb, "2m_temperature", "t2m RMSE (K)", title="2 m temperature")
    axb.legend(ncol=2, loc="upper left", fontsize=8.5, handlelength=1.1, columnspacing=0.9, borderaxespad=0.1)
    fig.text(5.0 / 10.4, 1 - 0.43 / 6.3, "b", fontsize=13, fontweight="bold", va="bottom")
    specs = [("mean_sea_level_pressure", "MSLP RMSE (Pa)", "mean sea level pressure"),
             ("10m_u_component_of_wind", "u10 RMSE (m s$^{-1}$)", "10 m zonal wind"),
             ("temperature_850", "T850 RMSE (K)", "850 hPa temperature")]
    for k, (var, yl, tt) in enumerate(specs):
        ax = fig_in(fig, 0.75 + k * 3.35, 3.5, 2.55, 2.0)
        bars(ax, var, yl, title=tt)
        fig.text((0.3 + k * 3.35) / 10.4, 1 - 3.3 / 6.3, "cde"[k], fontsize=13, fontweight="bold", va="bottom")
    fig.savefig("figures/fig4_terrain.png"); plt.close(fig)
    print("terrain written; land fractions", [round(f, 1) for f in frac])


# ---------------------------------------------------------------- case studies
CASES = [("2020-01-16 00:00:00", "2m_temperature", (70, 110, 10, 50), "RdYlBu_r", "K", 1.0, "2 m temperature\n16 Jan 2020, 00 UTC", "Tibetan Plateau, Himalaya"),
         ("2020-07-16 12:00:00", "specific_humidity_700", (60, 100, 0, 40), "YlGnBu", "g kg$^{-1}$", 1000.0, "700 hPa humidity\n16 Jul 2020, 12 UTC", "Indian monsoon, W. Ghats"),
         ("2020-09-05 12:00:00", "mean_sea_level_pressure", (115.5, 145.5, 10, 40), "viridis", "hPa", 0.01, "sea level pressure\n5 Sep 2020, 12 UTC", "Typhoon Haishen")]
COLS = ["x", "bilinear+proj", "edsr", "orolift", "truth"]


def crop(a, box):
    lo, hi, la0, la1 = box
    li = np.where((LON >= lo) & (LON <= hi))[0]; ji = np.where((LAT >= la0) & (LAT <= la1))[0]
    return a[np.ix_(ji, li)], [LON[li[0]] - .75, LON[li[-1]] + .75, LAT[ji[0]] - .75, LAT[ji[-1]] + .75]


def cases():
    cw, gap, ch = 1.5, 0.07, 1.5
    pitch = ch + 0.68
    fig = plt.figure(figsize=(10.4, 0.55 + 3 * pitch - 0.2))
    x0 = 0.5
    cv = list(D["case_vars"])
    for ri, (t, var, box, cmap, unit, sc, ttl, sub) in enumerate(CASES):
        y = 0.55 + ri * pitch
        k = cv.index(var)
        F = {m: D[f"case|{t}|{m}"][k] * sc for m in COLS if m != "x"}
        F["x"] = D[f"case|{t}|x"][k] * sc
        ref, ext = crop(F["truth"], box)
        lo, hi = np.percentile(ref, 1), np.percentile(ref, 99)
        # context map
        axc = fig_in(fig, x0, y, cw * 1.2, ch)
        axc.imshow(roll(F["truth"]), origin="lower", extent=EXT, cmap=cmap, aspect="auto", interpolation="nearest",
                   vmin=np.percentile(F["truth"], 1), vmax=np.percentile(F["truth"], 99))
        axc.contour(LONR, LAT, roll(LSM), levels=[0.5], colors="k", linewidths=0.25)
        b = [(box[0] + 360) % 360, box[1], box[2], box[3]]
        bx = ((box[0] + 180) % 360) - 180, ((box[1] + 180) % 360) - 180
        axc.add_patch(Rectangle((bx[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=S.ACCENT, lw=1.4))
        axc.set_xticks([]); axc.set_yticks([])
        axc.set_title("ERA5, global", fontsize=9.5, pad=3) if ri == 0 else None
        fig.text(0.01, 1 - (y - 0.03) / fig.get_size_inches()[1], "abc"[ri], fontsize=13, fontweight="bold", va="bottom")
        xs = x0 + cw * 1.2 + 0.2
        lat_box = (box[2], box[3])
        for ci, m in enumerate(COLS):
            ax = fig_in(fig, xs + ci * (cw + gap), y, cw, ch)
            if m == "x":
                lo_i = np.where((LON_LR >= box[0] - 3) & (LON_LR <= box[1] + 3))[0]; la_i = np.where((LAT_LR >= box[2] - 3) & (LAT_LR <= box[3] + 3))[0]
                im = ax.imshow(F["x"][np.ix_(la_i, lo_i)], origin="lower",
                               extent=[LON_LR[lo_i[0]] - 2.8125, LON_LR[lo_i[-1]] + 2.8125, LAT_LR[la_i[0]] - 2.8125, LAT_LR[la_i[-1]] + 2.8125],
                               cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            else:
                f, _ = crop(F[m], box)
                im = ax.imshow(f, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")
                if m != "truth":
                    r = np.sqrt(np.mean((f - ref) ** 2))
                    S.note(ax, f"RMSE {r:.3g}", fs=8, loc="bl")
            lo_i = np.where((LON >= box[0]) & (LON <= box[1]))[0]
            sl = roll(LSM)[:, :]
            ax.contour(LON, LAT, LSM, levels=[0.5], colors="k", linewidths=0.35) if True else None
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_edgecolor(S.ACCENT if ci > 0 or True else "k"); s.set_linewidth(1.2 if m == "orolift" else 0.6)
            if ri == 0:
                ax.set_title(S.LABEL[m] if m != "edsr" else "EDSR", fontsize=9.5, pad=3,
                             color=S.ACCENT if m == "orolift" else "k", fontweight="bold" if m == "orolift" else "normal")
        cax = fig_in(fig, xs, y + ch + 0.11, 5 * cw + 4 * gap, 0.09)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
        cb.ax.tick_params(labelsize=8, length=2); cb.outline.set_linewidth(0.5)
        cb.set_label(unit, fontsize=9, labelpad=1)
        fig.text((x0) / 10.4, 1 - (y + ch + 0.08) / fig.get_size_inches()[1], f"{ttl}\n{sub}", fontsize=8.5, va="top", ha="left", linespacing=1.3)
    fig.savefig("figures/fig1_case_t2m.png"); plt.close(fig)
    print("cases written")


if __name__ == "__main__":
    terrain(); cases()
