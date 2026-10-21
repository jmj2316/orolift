"""Restyled zonal-spectra figure (fig2) and consistency-certificate figure (fig5)."""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style as S
from orolift.varlist import VARS_69

d = np.load("figures/fig_spec.npz", allow_pickle=True)
names = list(d["names"]); spec = d["spec"]
VARS = [("2m_temperature", "2 m temperature"), ("geopotential_500", "500 hPa geopotential"),
        ("u_component_of_wind_850", "850 hPa zonal wind"), ("specific_humidity_700", "700 hPa humidity")]
ST = {"truth": ("-", 1.9), "bilinear": ("-", 1.1), "bilinear+proj": ("-", 1.1), "liif": ("-", 1.1),
      "edsr": ("-", 1.1), "orolift": ("-", 1.6)}
kk = np.arange(spec.shape[-1])

fig, ax = plt.subplots(2, 4, figsize=(10.2, 5.0), gridspec_kw=dict(height_ratios=[1.45, 1], hspace=0.12, wspace=0.36), sharex="col")
for j, (v, ttl) in enumerate(VARS):
    i = VARS_69.index(v); t = spec[0, i]
    for n in ["truth"] + S.METHODS:
        m = names.index(n); ls, lw = ST[n]
        ax[0, j].loglog(kk[1:], spec[m, i, 1:], color=S.PAL[n], ls=ls, lw=lw, label=S.LABEL[n], zorder=5 if n in ("truth", "orolift") else 3)
        if n != "truth":
            ax[1, j].semilogx(kk[1:], spec[m, i, 1:] / t[1:], color=S.PAL[n], ls=ls, lw=lw, zorder=5 if n == "orolift" else 3)
    for a in ax[:, j]:
        a.axvline(32, color="0.4", lw=0.8, ls=":")
        a.axvspan(32, kk[-1] + 2, color="0.5", alpha=0.07, lw=0)
    ax[0, j].set_title(ttl, fontsize=10)
    ax[1, j].axhline(1, color="0.15", lw=0.9)
    ax[1, j].set_ylim(0, 1.6); ax[1, j].set_xlim(1, kk[-1] + 2)
    ax[1, j].set_xlabel("zonal wavenumber $k$")
    ax[1, j].text(0.69, 0.97, "5.625° Nyquist →", transform=ax[1, j].transAxes, fontsize=7.5, color="0.35", va="top", ha="right")
    S.tag(ax[0, j], "abcd"[j], dx=0.0, dy=1.1, fs=13)
    ax[0, j].xaxis.set_tick_params(labelbottom=False)
ax[0, 0].set_ylabel("zonal power"); ax[1, 0].set_ylabel("power ratio to ERA5")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=9)
fig.savefig("figures/fig2_spectra_new.png"); plt.close(fig)

# ----------------------------------------------------------------- certificate
fig, ax = plt.subplots(figsize=(3.9, 3.0))
vals = [9.0e-2, 6.3e-8, 7.3e-13]
lbl = ["no projection\n(unconstrained)", "projection\nfp32", "projection\nfp64"]
cols = [S.PAL["bilinear+proj"], S.ACCENT, S.ACCENT]
ax.bar(range(3), vals, color=cols, width=0.62, edgecolor="0.2", lw=0.6)
ax.set_yscale("log"); ax.set_ylim(1e-14, 1); ax.set_xticks(range(3), lbl, fontsize=8.5)
ax.set_ylabel(r"$\|A\hat y - x\|$  (normalized units)")
for i, v in enumerate(vals):
    ax.text(i, v * 2.4, f"{v:.1e}".replace("e-0", "×10$^{-").replace("e-", "×10$^{-") + "}$", ha="center", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Consistency residual of the 1.5° prediction", fontsize=9.5)
fig.savefig("figures/fig5_certificate_new.png"); plt.close(fig)
print("ok")
