"""Restyled ensemble-diagnostics figure (fig12) from the saved eval_flow data."""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style as S
from orolift.grids import LAT_HR, LON_HR
from orolift.varlist import VARS_69, LEVELS, ATMOS_PREFIXES

d = np.load("runs/eval/flow_v2_r_figdata.npz")
ss_all, ranks, kk, sp, sig_q, lsm = (d[k] for k in ["ss_all", "ranks", "kk", "sp", "sig_q", "lsm"])
M = ranks.shape[1] - 1
KEY3 = ["2m_temperature", "mean_sea_level_pressure", "geopotential_500"]
NAME = {"geopotential": "Geopotential", "temperature": "Temperature", "u_component_of_wind": "Zonal wind",
        "v_component_of_wind": "Meridional wind", "specific_humidity": "Humidity"}
cols = ["#1D2B55", "#3F6BBE", "#7F9FD3", "#D64550", "#E8A33D"]
RT = {"2m_temperature": "2 m temperature", "mean_sea_level_pressure": "Sea-level pressure",
      "geopotential_500": "500 hPa geopotential"}

fig = plt.figure(figsize=(10.2, 7.4))
gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
for p, c in zip(ATMOS_PREFIXES, cols):
    vals = [ss_all[VARS_69.index(f"{p}_{l}")] for l in LEVELS]
    ax.plot(vals, LEVELS, "o-", ms=3, lw=1.2, color=c, label=NAME.get(p, p))
ax.axvline(1.0, color="0.2", lw=0.9, ls="--")
ax.invert_yaxis(); ax.set_yscale("log"); ax.set_yticks(LEVELS, [str(l) for l in LEVELS], fontsize=7)
ax.minorticks_off(); ax.set_xlabel("spread / skill"); ax.set_ylabel("pressure (hPa)")
ax.set_xlim(0.885, 1.2); ax.set_xticks([0.9, 1.0])
ax.legend(fontsize=7, loc="lower right", handlelength=1.3, borderaxespad=0.2, frameon=False)
ax.set_title("Calibration", fontsize=9.5); S.tag(ax, "a", dx=0.0, dy=1.1)

for j, v in enumerate(KEY3):
    axr = fig.add_subplot(gs[0, 1 + j])
    h = ranks[VARS_69.index(v)]; h = h / h.sum()
    axr.bar(np.arange(M + 1), h, color=S.PAL["liif"], edgecolor="0.2", lw=0.4, width=0.85)
    axr.axhline(1 / (M + 1), color=S.ACCENT, lw=1.2, ls="--")
    axr.set_ylim(0, h.max() * 1.15)
    axr.set_title(RT[v], fontsize=9.5); axr.set_xlabel("rank of truth in ensemble")
    axr.set_yticks([]); axr.spines[["top", "right", "left"]].set_visible(False)
    if j == 0:
        axr.set_ylabel("frequency"); S.tag(axr, "b", dx=0.0, dy=1.1)

axs = fig.add_subplot(gs[1, 0:2])
for i, lab, c, ls, lw in [(0, "ERA5 1.5°", S.PAL["truth"], "-", 1.9), (1, "Deterministic OroLIFT", S.PAL["liif"], "-", 1.2),
                          (2, "Ensemble mean", S.PAL["bilinear+proj"], "--", 1.2), (3, "Single member", S.ACCENT, "-", 1.5)]:
    axs.loglog(kk[1:], sp[i][1:], color=c, ls=ls, lw=lw, label=lab)
axs.axvline(32, color="0.4", lw=0.8, ls=":"); axs.axvspan(32, kk[-1] + 2, color="0.5", alpha=0.07, lw=0)
axs.set_xlim(1, kk[-1] + 2)
axs.set_xlabel("zonal wavenumber $k$"); axs.set_ylabel("zonal power")
axs.legend(fontsize=8, loc="lower left", frameon=False)
axs.set_title("850 hPa zonal wind: members restore the spectral tail", fontsize=9.5)
S.tag(axs, "c", dx=0.0, dy=1.1)

axm = fig.add_subplot(gs[1, 2:4])
lon = np.where(LON_HR > 180, LON_HR - 360, LON_HR); o = np.argsort(lon)
Z = sig_q[:, o] * 1e3; Lm = lsm[:, o]; lo = lon[o]
im = axm.pcolormesh(lo, LAT_HR, Z, cmap="magma_r", shading="nearest", rasterized=True, antialiased=False)
axm.contour(lo, LAT_HR, Lm, levels=[0.5], colors="w", linewidths=0.4)
axm.set_xlim(-180, 180); axm.set_ylim(-90, 90); axm.set_aspect("equal")
axm.set_xticks([-120, 0, 120], ["120°W", "0°", "120°E"]); axm.set_yticks([-60, 0, 60], ["60°S", "0°", "60°N"])
cb = fig.colorbar(im, ax=axm, shrink=0.8, pad=0.02); cb.set_label(r"g kg$^{-1}$")
axm.set_title("Conditional spread of 700 hPa humidity", fontsize=9.5); S.tag(axm, "d", dx=0.0, dy=1.1)
fig.savefig("figures/fig12_flow_new.png"); plt.close(fig)
print("ok")
