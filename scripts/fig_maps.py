"""Error-map grids in the style of method x metric panels: RMSE / anomaly-R2 / MBE, annotated domain means."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from orolift.data import static_features

D = np.load("figures/fig_stats.npz", allow_pickle=True)
M, V = list(D["methods"]), list(D["stat_vars"])
LAT = np.linspace(-90, 90, 121)
LON = np.arange(240) * 1.5
W = np.cos(np.deg2rad(LAT)); W = W / W.mean()
SH = 120                                   # roll 0..360 -> -180..180
LONR = np.roll(LON, -SH); LONR = np.where(LONR >= 180, LONR - 360, LONR)
hr, _ = static_features("cpu")
ORO = hr[0].numpy() * 1000.0; DET = hr[1].numpy() * 1000.0; LSM = hr[3].numpy()
roll = lambda a: np.roll(a, -SH, axis=-1)
EXT = [-180 + 0.0, 180 - 1.5, -90, 90]
EXT = [LONR.min() - .75, LONR.max() + .75, -90.75, 90.75]


def fig_in(fig, x, y, w, h):
    W_, H_ = fig.get_size_inches()
    return fig.add_axes([x / W_, 1 - (y + h) / H_, w / W_, h / H_])


def showmap(ax, a, cmap, vmin, vmax, coast=True):
    im = ax.imshow(roll(a), origin="lower", extent=EXT, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    if coast:
        ax.contour(LONR, LAT, roll(LSM), levels=[0.5], colors="k", linewidths=0.3, alpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.5)
    return im


def stat_fields(v):
    vi = V.index(v)
    sse, mbe, sst = D["sse"][:, vi], D["mbe"][:, vi], D["sst"][vi]
    rmse = np.sqrt(sse)
    r2 = 1 - sse / sst[None]
    wm = lambda a: float((a * W[:, None]).mean())
    means = dict(rmse=[np.sqrt(wm(sse[m])) for m in range(5)],
                 r2=[1 - wm(sse[m]) / wm(sst) for m in range(5)],
                 mbe=[wm(np.abs(mbe[m])) for m in range(5)])
    prof = np.sqrt(sse.mean(-1))                     # (5, 121) zonal RMSE
    return rmse, r2, mbe, means, prof


def grid(vars_, unit, rmax, mmax, fmt, out, with_b=True, scale=1.0, ptitle=None):
    cw, gap, mh = 1.72, 0.08, 0.86
    rows = len(vars_) if isinstance(vars_, list) else 3
    fig = plt.figure(figsize=(10.4, 0.5 + 3 * (mh + .12) + (1.85 if with_b else 0.2)))
    x0, y0 = 0.78, 0.45
    v = vars_[0]
    rmse, r2, mbe, means, prof = stat_fields(v)
    spec = [("RMSE (%s)" % unit, rmse * scale, "YlOrRd", 0, rmax, means["rmse"], "RMSE", "max"),
            ("anomaly $R^2$", r2, "YlGnBu", 0, 1, means["r2"], "R²", "both"),
            ("MBE (%s)" % unit, mbe * scale, "RdBu_r", -mmax, mmax, means["mbe"], "|MBE|", "both")]
    for ri, (lab, fld, cm, lo, hi, mean, mname, ext) in enumerate(spec):
        y = y0 + ri * (mh + 0.12)
        for ci in range(5):
            ax = fig_in(fig, x0 + ci * (cw + gap), y, cw, mh)
            im = showmap(ax, fld[ci], cm, lo, hi)
            if ri == 0:
                ax.set_title(S.LABEL[M[ci]] if M[ci] != "edsr" else "EDSR", fontsize=9.5, pad=3,
                             color=S.ACCENT if M[ci] == "orolift" else "k",
                             fontweight="bold" if M[ci] == "orolift" else "normal")
            mv = mean[ci] * (scale if ri != 1 else 1)
            S.note(ax, "mean " + (fmt[ri] % mv if ri != 2 else "|MBE| " + fmt[ri] % mv).replace("mean |MBE| ", "|MBE| "), fs=8, loc="bl")
            if ci == 0:
                ax.set_ylabel(lab, fontsize=9, labelpad=3)
        cax = fig_in(fig, x0 + 5 * (cw + gap) + 0.02, y + 0.06, 0.11, mh - 0.12)
        cb = fig.colorbar(im, cax=cax, extend=ext)
        cb.ax.tick_params(labelsize=8, length=2)
        cb.outline.set_linewidth(0.5)
    fig.text(0.012, 1 - 0.45 / fig.get_size_inches()[1] + 0.012, "a", fontsize=13, fontweight="bold", va="bottom")
    if with_b:
        yb = y0 + 3 * (mh + .12) + 0.42
        fig.text(0.012, 1 - (yb - 0.1) / fig.get_size_inches()[1], "b", fontsize=13, fontweight="bold", va="bottom")
        a1 = fig_in(fig, x0, yb, cw, mh); i1 = showmap(a1, np.where(LSM > 0.5, ORO, np.nan), "terrain", 0, 5000)
        a1.set_title("orography (m)", fontsize=9.5, pad=3)
        a2 = fig_in(fig, x0 + cw + gap, yb, cw, mh); i2 = showmap(a2, DET, "PuOr_r", -800, 800)
        a2.set_title("sub-grid detail (m)", fontsize=9.5, pad=3)
        for a, i, ext_ in ((a1, i1, "max"), (a2, i2, "both")):
            p = a.get_position()
            c = fig.add_axes([p.x0, p.y0 - 0.115 / fig.get_size_inches()[1] - 0.012, p.width, 0.012])
            cb = fig.colorbar(i, cax=c, orientation="horizontal", extend=ext_)
            cb.ax.tick_params(labelsize=8, length=2); cb.outline.set_linewidth(0.5)
        fig.text(0.012 + (x0 + 2 * (cw + gap) + 0.55) / fig.get_size_inches()[0] - 0.012,
                 1 - (yb - 0.1) / fig.get_size_inches()[1], "c", fontsize=13, fontweight="bold", va="bottom")
        ap = fig_in(fig, x0 + 2 * (cw + gap) + 0.75, yb - 0.02, 3 * cw + 2 * gap - 0.75 + 0.25, mh + 0.2)
        for m in range(5):
            ap.plot(LAT, prof[m] * scale, color=S.PAL[M[m]], lw=2.0 if M[m] == "orolift" else 1.3,
                    label=S.LABEL[M[m]] if M[m] != "edsr" else "EDSR", zorder=5 if M[m] == "orolift" else 3)
        ap.set_xlim(-90, 90); ap.set_xticks(range(-90, 91, 30)); ap.set_xticklabels([f"{abs(t)}°{'S' if t<0 else 'N' if t>0 else ''}" for t in range(-90, 91, 30)])
        ap.set_ylabel(f"zonal RMSE ({unit})", fontsize=9)
        ap.set_ylim(0, ap.get_ylim()[1] * 1.35)
        for s in ("top", "right"): ap.spines[s].set_visible(False)
        ap.legend(ncol=3, loc="upper center", fontsize=8.5, handlelength=1.3, columnspacing=1.0)
    fig.savefig(out); plt.close(fig)
    print("wrote", out, [round(x, 3) for x in means["rmse"]])


def grid_vars(vars_, titles, rmaxs, units, scales, fmts, out):
    cw, gap, mh = 1.72, 0.08, 0.86
    n = len(vars_)
    fig = plt.figure(figsize=(10.4, 0.5 + n * (mh + .12)))
    x0, y0 = 0.9, 0.45
    for ri, v in enumerate(vars_):
        rmse, _, _, means, _ = stat_fields(v)
        y = y0 + ri * (mh + 0.12)
        for ci in range(5):
            ax = fig_in(fig, x0 + ci * (cw + gap), y, cw, mh)
            im = showmap(ax, rmse[ci] * scales[ri], "YlOrRd", 0, rmaxs[ri])
            if ri == 0:
                ax.set_title(S.LABEL[M[ci]] if M[ci] != "edsr" else "EDSR", fontsize=9.5, pad=3,
                             color=S.ACCENT if M[ci] == "orolift" else "k",
                             fontweight="bold" if M[ci] == "orolift" else "normal")
            S.note(ax, "mean " + fmts[ri] % (means["rmse"][ci] * scales[ri]), fs=8, loc="bl")
            if ci == 0:
                ax.set_ylabel(f"{titles[ri]}\nRMSE ({units[ri]})", fontsize=9, labelpad=3)
        cax = fig_in(fig, x0 + 5 * (cw + gap) + 0.02, y + 0.06, 0.11, mh - 0.12)
        cb = fig.colorbar(im, cax=cax, extend="max"); cb.ax.tick_params(labelsize=8, length=2); cb.outline.set_linewidth(0.5)
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    grid(["2m_temperature"], "K", 3.0, 1.5, ["%.3f", "%.3f", "%.3f"], "figures/fig_maps_t2m.png")
    grid_vars(["mean_sea_level_pressure", "10m_u_component_of_wind", "temperature_850", "geopotential_500"],
              ["MSLP", "10 m u-wind", "T850", "$Z_{500}$"], [300, 3.0, 2.0, 150],
              ["Pa", "m s$^{-1}$", "K", "m$^2$ s$^{-2}$"], [1, 1, 1, 1],
              ["%.1f", "%.3f", "%.3f", "%.1f"], "figures/fig_maps_other.png")
