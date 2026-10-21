"""Empirical range/null-space decomposition of the downscaling error (runs/eval/decomp.json)."""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style as S
from orolift.varlist import VARS_69, LEVELS

d = json.load(open("runs/eval/decomp.json")); V = d["vars"]; assert V == VARS_69
A = {k: np.array(v) for k, v in d["A"].items()}
B = {k: np.array(v) for k, v in d["B"].items()}
Cc = {k: np.array(v) for k, v in d["C"].items()}
LAGS = [0, 1, 4, 12]; LAGLAB = ["analysis\n(lag 0)", "6 h", "1 d", "3 d"]
MODS = ["liif", "edsr", "orolift"]

fig = plt.figure(figsize=(10.4, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

# (a) range-space share of the squared error of each unconstrained output
rng = np.random.default_rng(0)
order = [("bilinear", "Bilinear"), ("liif", "LIIF"), ("edsr", "EDSR"), ("orolift", "OroLIFT"), ("noproj", "OroLIFT,\nno proj. in training")]
for j, (k, lab) in enumerate(order):
    sh = A[k][1] / A[k][0]
    col = S.PAL.get(k, "0.55")
    ax[0].scatter(j + rng.uniform(-0.18, 0.18, len(sh)), sh, s=9, color=col, alpha=0.65, lw=0, zorder=3)
    m = np.median(sh)
    ax[0].hlines(m, j - 0.32, j + 0.32, color="0.1", lw=1.6, zorder=4)
    ax[0].text(j, 1.035, f"{m:.2f}", ha="center", fontsize=8.5)
ax[0].set_xticks(range(len(order)), [o[1] for o in order], fontsize=8)
ax[0].set_ylim(0, 1.1); ax[0].set_yticks(np.arange(0, 1.01, 0.25))
ax[0].set_ylabel("range-space share of squared error")
ax[0].set_title("Output before projection, 69 variables", fontsize=9.5)
ax[0].spines[["top", "right"]].set_visible(False)

# (b) the same share for the bilinear interpolant, by variable and level
cls = [("geopotential", "$Z$"), ("temperature", "$T$"), ("u_component_of_wind", "$u$"),
       ("v_component_of_wind", "$v$"), ("specific_humidity", "$q$")]
cc = ["#1D2B55", "#D64550", "#3F6BBE", "#7F9FD3", "#2A9D8F"]
sh_b = A["bilinear"][1] / A["bilinear"][0]
for (p, lab), c in zip(cls, cc):
    y = [sh_b[V.index(f"{p}_{l}")] for l in LEVELS]
    ax[1].plot(y, LEVELS, "-o", ms=3, color=c, label=lab, lw=1.2)
for v, lab, mk, dx in [("2m_temperature", "2 m $T$", "s", 7), ("mean_sea_level_pressure", "MSLP", "D", 7)]:
    ax[1].plot(sh_b[V.index(v)], 1190, mk, color="0.2", ms=4.5)
    ax[1].annotate(lab, (sh_b[V.index(v)], 1190), xytext=(dx, -1), textcoords="offset points", ha="left", va="center", fontsize=7.5)
ax[1].set_yscale("log"); ax[1].invert_yaxis(); ax[1].set_ylim(1300, 42)
ax[1].set_yticks(LEVELS[::2], [str(l) for l in LEVELS[::2]]); ax[1].minorticks_off()
ax[1].set_ylabel("pressure level (hPa)"); ax[1].set_xlabel("range-space share of bilinear error")
ax[1].legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, columnspacing=1.0, handlelength=1.3)
ax[1].set_title("What the input already determines", fontsize=9.5, pad=22)
ax[1].spines[["top", "right"]].set_visible(False)

# (c) lagged analysis: range term is fixed by the input, null term is what the model changes
cases = [("bilinear+proj", "Bilinear + proj."), ("orolift", "OroLIFT")]
x0 = np.arange(len(LAGS)) * 3.0
for j, (k, lab) in enumerate(cases):
    for i, lag in enumerate(LAGS):
        a = B[f"{lag}|{k}"]
        r, n = a[1].mean(), a[2].mean()
        xx = x0[i] + j * 1.05
        ax[2].bar(xx, r, width=1.0, color="0.78", edgecolor="0.3", lw=0.5, label="range space (set by input)" if (i == 0 and j == 0) else None)
        ax[2].bar(xx, n, width=1.0, bottom=r, color=S.PAL[k], edgecolor="0.3", lw=0.5,
                  label=f"null space, {lab}" if i == 0 else None)
ax[2].set_xticks(x0 + 0.52, LAGLAB, fontsize=8)
ax[2].set_ylabel("mean squared error (normalized units)")
ax[2].set_xlabel("age of the 5.625° field (persistence)")
ax[2].legend(loc="upper left", fontsize=7.5)
ax[2].set_title("Stale input: error passes through", fontsize=9.5)
ax[2].spines[["top", "right"]].set_visible(False)

# (d) relative gain over bilinear + proj. as the input's range-space error grows
MK = {"liif": "^", "edsr": "s", "orolift": "o"}
for k in MODS:
    xb, yb, xl, yl = [], [], [], []
    for al in sorted(float(s_.split("|")[0]) for s_ in Cc if s_.endswith("|orolift")):
        b_ = Cc[f"{al}|bilinear+proj"]; m = Cc[f"{al}|{k}"]
        xb.append(m[1].mean()); yb.append(100 * (1 - m[0].mean() / b_[0].mean()))
    for lag in LAGS[1:]:
        b_ = B[f"{lag}|bilinear+proj"]; m = B[f"{lag}|{k}"]
        xl.append(m[1].mean()); yl.append(100 * (1 - m[0].mean() / b_[0].mean()))
    b0 = B["0|bilinear+proj"]; m0 = B[f"0|{k}"]
    xb = [2e-4] + xb; yb = [100 * (1 - m0[0].mean() / b0[0].mean())] + yb
    ax[3].plot(xb, yb, "-", marker=MK[k], ms=3.6, color=S.PAL[k], lw=1.2, label=S.LABEL[k])
    ax[3].plot(xl, yl, "--", marker=MK[k], ms=3.6, color=S.PAL[k], lw=1.0, mfc="white")
for lag, lab in zip(LAGS[1:], LAGLAB[1:]):
    b_ = B[f"{lag}|bilinear+proj"]; m = B[f"{lag}|orolift"]
    ax[3].annotate(lab, (m[1].mean(), 100 * (1 - m[0].mean() / b_[0].mean())), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7.5)
ax[3].axhline(0, color="0.2", lw=0.8); ax[3].set_ylim(-30, 46); ax[3].set_yticks([0, 10, 20, 30, 40])
ax[3].set_xscale("log"); ax[3].set_xlim(1.2e-4, 0.9)
ax[3].set_xticks([2e-4, 1e-3, 1e-2, 1e-1], ["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
ax[3].set_xlabel("range-space error of the input (normalized MSE)")
ax[3].set_ylabel("reduction in total squared error\nvs bilinear + proj. (%)")
h, l = ax[3].get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], color="0.3", ls="-", lw=1.2), Line2D([], [], color="0.3", ls="--", lw=1.0)]
l += ["input error scaled (1-day difference $\\times\\,\\alpha$)", "input aged by persistence"]
ax[3].legend(h, l, fontsize=7.2, loc="lower left")
ax[3].set_title("Gain over the projected interpolant", fontsize=9.5)
ax[3].spines[["top", "right"]].set_visible(False)

for a, t in zip(ax, "abcd"):
    S.tag(a, t, dx=-0.0, dy=1.06 if t != "b" else 1.12, fs=13)
fig.savefig("figures/fig19_decomp_new.png"); plt.close(fig)
print("ok")
