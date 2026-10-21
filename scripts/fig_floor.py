"""Restyled fig17: irreducible fraction vs capacity and data, plotted from runs/floor.json."""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style as S
from orolift.varlist import VARS_69

out = json.load(open("runs/floor.json"))
br = np.array(json.load(open("runs/eval/baseline_test_synthetic.json"))["rmse"])
CAP = [("sc_dim48", "dim 48"), ("cmp_orolift", "dim 96"), ("sc_dim192", "dim 192")]
DAT = [("sc_data8", "1/8 data"), ("sc_data2", "1/2 data"), ("cmp_orolift", "full data")]
KEY = [("2m_temperature", "2 m temperature"), ("mean_sea_level_pressure", "Sea-level pressure"),
       ("geopotential_500", "500 hPa geopotential"), ("temperature_850", "850 hPa temperature"),
       ("u_component_of_wind_850", "850 hPa zonal wind"), ("specific_humidity_700", "700 hPa humidity")]
C_CAP, C_DAT = S.PAL["liif"], "#E8A33D"

fig, ax = plt.subplots(2, 3, figsize=(10.2, 6.4), gridspec_kw=dict(hspace=0.62, wspace=0.28))
for k, (v, ttl) in enumerate(KEY):
    a = ax.flat[k]; i = VARS_69.index(v); xs = np.arange(3)
    for pts, c, mk in [(CAP, C_CAP, "o"), (DAT, C_DAT, "s")]:
        f = [out[r]["f"][i] for r, _ in pts]; q = [out[r]["rmse"][i] / br[i] for r, _ in pts]
        a.plot(xs, f, mk + "-", color=c, lw=1.8, ms=5.5, zorder=4)
        a.plot(xs, q, mk + ":", color=c, lw=1.4, ms=4.5, alpha=0.55, zorder=3)
    a.set_ylim(0, 1.05); a.set_xlim(-0.3, 2.3); a.set_xticks(xs, [""] * 3)
    for x, (_, l1), (_, l2) in zip(xs, CAP, DAT):
        a.annotate(l1, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -6), textcoords="offset points",
                   ha="center", va="top", fontsize=7.8, color=C_CAP)
        a.annotate(l2, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -17), textcoords="offset points",
                   ha="center", va="top", fontsize=7.8, color="#B8761A")
    a.set_title(ttl, fontsize=9.5); a.spines[["top", "right"]].set_visible(False)
    S.tag(a, "abcdef"[k], dx=0.0, dy=1.1)
    if k % 3 == 0:
        a.set_ylabel("fraction of baseline error")
h = [Line2D([], [], color=C_CAP, marker="o", lw=1.8, label="capacity axis (trunk width)"),
     Line2D([], [], color=C_DAT, marker="s", lw=1.8, label="data axis (training samples)"),
     Line2D([], [], color="0.3", lw=1.8, label=r"$f_{\mathrm{irr}}$ (variance-head estimate)"),
     Line2D([], [], color="0.3", lw=1.4, ls=":", alpha=0.7, label="mean-model RMSE / baseline RMSE")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=8.6, columnspacing=1.4)
fig.savefig("figures/fig17_floor_new.png"); plt.close(fig)
print("ok")
