"""Central-pressure error of every method on the same 25 distinct warm-core cyclones of 2020."""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

st = json.load(open("runs/eval/typhoon_cmp_storms.json")); R = json.load(open("runs/eval/typhoon_cmp.json"))
t = np.array([r["true"] for r in st])
get = lambda k: np.array([r[k]["basin"] for r in st])
M = [("input", "x", "^", "5.625° input"), ("bilinear+proj", "bilinear+proj", "s", "Bilinear + proj."),
     ("edsr", "edsr", "D", "EDSR"), ("liif", "liif", "v", "LIIF"), ("orolift", "orolift", "o", "OroLIFT")]
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9), gridspec_kw=dict(wspace=0.27))
lim = (t.min() - 3, 1002)
ax[0].plot(lim, lim, "-", color="0.2", lw=0.9, zorder=1)
x = np.linspace(*lim, 50)
for k, c, mk, lab in M:
    p = get(k); mae = R[k]["mae"]
    ax[0].scatter(t, p, s=(20 if k not in ("orolift",) else 26), color=S.PAL[c], marker=mk, alpha=0.9 if k in ("input", "bilinear+proj", "orolift") else 0.55,
                  lw=0, zorder=4 if k == "orolift" else 3, label=f"{lab} (MAE {mae:.1f})")
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("ERA5 1.5° central pressure (hPa)"); ax[0].set_ylabel("reconstructed central pressure (hPa)")
ax[0].legend(loc="lower right", fontsize=7.8, handletextpad=0.2, labelspacing=0.3)
ax[0].set_title("25 distinct 2020 cyclones", fontsize=9.5)
ax[1].axhline(0, color="0.2", lw=0.9, zorder=1)
for k, c, mk, lab in M[1:]:
    e = get(k) - t; a, b = np.polyfit(t, e, 1)
    ax[1].scatter(t, e, s=(20 if k != "orolift" else 26), color=S.PAL[c], marker=mk, alpha=0.9 if k in ("bilinear+proj", "orolift") else 0.5,
                  lw=0, zorder=4 if k == "orolift" else 3)
    ax[1].plot(x, a * x + b, color=S.PAL[c], lw=1.3 if k == "orolift" else 1.0, zorder=2,
               label=f"{lab}: slope {a:+.2f}")
ax[1].set_xlim(lim)
ax[1].set_xlabel("ERA5 1.5° central pressure (hPa)"); ax[1].set_ylabel("error (hPa)")
ax[1].legend(loc="upper right", fontsize=7.8, handletextpad=0.4, labelspacing=0.3)
ax[1].set_title("Error against storm depth (OLS fits)", fontsize=9.5)
for a, tt in zip(ax, "ab"):
    a.spines[["top", "right"]].set_visible(False); S.tag(a, tt, dx=0.0, dy=1.08)
fig.savefig("figures/fig20_tc_cmp.png"); print("ok")
