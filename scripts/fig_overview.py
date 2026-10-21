"""Framework overview: (1) training with exact coarsening, (2) prediction, in the 3-D slab style."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

D = np.load("figures/arch_inputs.npz")
NAVY, BLUE, ORNG, GREEN, RED = "#2E4A6B", "#2C6FBB", "#E8873A", "#2E8B6E", "#B23A48"
ZC, ZF = "#EAF1FA", "#FDF1E6"
fig = plt.figure(figsize=(10.0, 6.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 63); ax.axis("off")
FS = 9.5


def shade(c, f):
    c = np.array(matplotlib.colors.to_rgb(c))
    return tuple(np.clip(c * f if f < 1 else c + (1 - c) * (f - 1), 0, 1))


def T(x, y, s, fs=FS, c="0.12", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=6, linespacing=1.25, **kw)


def box(x, y, w, h, t="", fc="#ECECEC", tc="0.12", fs=FS, r=0.8, ec="0.4", lw=0.9, ls="-", z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if t: T(x + w / 2, y + h / 2, t, fs=fs, c=tc)


def arrow(p, q, rad=0.0, c="0.25", lw=1.3, ls="-", ms=12, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=c, lw=lw, ls=ls, mutation_scale=ms,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def slab(x, y, w, h, c, dep=0.9):
    ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="0.2", lw=0.8, zorder=3))
    ax.add_patch(Polygon([(x, y + h), (x + dep, y + h + dep), (x + w + dep, y + h + dep), (x + w, y + h)], fc=shade(c, 1.45), ec="0.2", lw=0.8, zorder=3))
    ax.add_patch(Polygon([(x + w, y), (x + w + dep, y + dep), (x + w + dep, y + h + dep), (x + w, y + h)], fc=shade(c, 0.78), ec="0.2", lw=0.8, zorder=3))


def img(a, x, y, w, h, cmap, vmin=None, vmax=None, frames=0, ec="0.3", ls="-"):
    for k in range(frames, 0, -1):
        ax.add_patch(Rectangle((x + 0.8 * k, y + 0.8 * k), w, h, fc="white", ec=ec, lw=0.8, zorder=2 + 0.01 * (5 - k)))
    ax.imshow(a, extent=[x, x + w, y, y + h], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", zorder=3.5, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=0.9, ls=ls, zorder=4))


def zone(y0, y1, fc):
    ax.add_patch(FancyBboxPatch((1, y0), 98, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="0.7", lw=0.9, zorder=0))


def network(yc, frozen):
    for k in range(5):
        slab(46 + k * 2.4, yc - 5.0, 1.8, 10.0, NAVY if 0 < k < 4 else "#5B7FA6")
    T(52.0, yc - 8.0, "SAFM trunk on the\n5.625° grid", fs=9)
    arrow((58.6, yc), (62.0, yc), lw=1.4, c=NAVY)
    box(62.2, yc - 6.0, 8.8, 12.0, "", fc="#F9DFC6", ec="0.35")
    T(66.6, yc + 1.5, "implicit\ndecoder", fs=9)
    T(66.6, yc - 3.2, "queried on\nthe 1.5° grid", fs=8.5, c="0.35", style="italic")
    arrow((71.2, yc), (74.4, yc), lw=1.4, c=NAVY)
    box(74.6, yc - 6.0, 5.6, 12.0, "", fc=RED, ec="0.25", z=4)
    T(77.4, yc, "$\\Pi_A$", fs=14, c="white", fontweight="bold", rotation=90)
    T(77.4, yc - 8.0, "exact\nprojection", fs=9, c=RED)
    T(56.5, yc + 8.3, "weights frozen" if frozen else "trainable (1.83 M)", fs=8.8, c="0.35", style="italic")


# ---------------------------------------------------------------- stage 1: training
zone(32.2, 61.6, ZC)
T(2.3, 59.3, "1", fs=15, ha="left", fontweight="bold", c=BLUE)
T(5.3, 59.3, "Training: reanalysis pairs related by an exact operator", fs=11, ha="left", fontweight="bold", c=BLUE)
yc = 46.0


def inputs(yc, label_oro, label_x):
    img(D["x_t2m"], 33.0, yc - 1.4, 8.0, 4.0, "RdYlBu_r", frames=2)
    T(37.6, yc - 4.6, label_x, fs=9)
    img(D["oro_hr"], 33.0, yc + 5.6, 8.0, 4.0, "terrain", vmin=0, vmax=4500)
    T(32.2, yc + 7.6, label_oro, fs=8.8, ha="right")
    arrow((43.4, yc + 0.8), (45.6, yc + 0.8), lw=1.2)
    arrow((41.6, yc + 7.6), (46.6, yc + 5.6), rad=-0.15, lw=1.2)


img(D["truth_t2m"], 3.0, yc - 3.3, 13.0, 6.6, "RdYlBu_r", ec="0.3")
T(9.5, yc - 6.6, "ERA5 1.5°\ntarget $y$", fs=9)
arrow((17.0, yc), (19.8, yc), lw=1.4)
box(20.0, yc - 4.2, 9.2, 8.4, "", fc="#E3ECF7", ec=NAVY, lw=1.2)
T(24.6, yc + 1.3, "coarsen", fs=9.2); T(24.6, yc - 1.7, "$A=A_{lat}\\!\\otimes\\!A_{lon}$", fs=9.2)
arrow((29.8, yc + 0.3), (32.4, yc + 0.5), lw=1.3)
inputs(yc, "1.5° orography\n+ sub-grid detail", "$x=Ay$  5.625°")
network(yc, False)
arrow((81.4, yc), (84.2, yc), lw=1.4)
img(D["pred_t2m"], 84.4, yc - 3.3, 13.0, 6.6, "RdYlBu_r", ec=RED)
T(91.0, yc - 6.2, "prediction $\\hat y$", fs=9, c=RED)
yl = yc - 11.3
ax.plot([9.5, 9.5], [yc - 9.0, yl + 0.3], color="0.35", lw=1.1, ls=(0, (4, 3)), zorder=1)
ax.plot([91.0, 91.0], [yc - 8.2, yl + 0.3], color="0.35", lw=1.1, ls=(0, (4, 3)), zorder=1)
ax.plot([9.5, 91.0], [yl, yl], color="0.35", lw=1.1, ls=(0, (4, 3)), zorder=1)
ax.plot([9.5], [yc - 8.6], marker="^", ms=5, color="0.35", zorder=2); ax.plot([91.0], [yc - 7.8], marker="^", ms=5, color="0.35", zorder=2)
ax.text(50.5, yl, "latitude-weighted MSE loss", fontsize=9, color="0.25", style="italic", ha="center", va="center", zorder=5,
        bbox=dict(fc=ZC, ec="none", pad=2))

# ---------------------------------------------------------------- stage 2: prediction
zone(1.2, 30.4, ZF)
T(2.3, 28.1, "2", fs=15, ha="left", fontweight="bold", c="#B8631F")
T(5.3, 28.1, "Prediction: any 5.625° field in, a 1.5° field out that averages back to it", fs=11, ha="left", fontweight="bold", c="#B8631F")
yc = 14.0
box(3.0, yc - 4.0, 15.0, 8.0, "", fc="white", ec="0.45", lw=1.0, ls=(0, (4, 3)))
T(10.5, yc + 1.4, "5.625° reanalysis", fs=9.2); T(10.5, yc - 1.4, "or ML forecast", fs=9.2)
T(10.5, yc - 6.8, "forecast input:\nnot tested here", fs=8.5, c="0.4", style="italic")
arrow((18.4, yc), (32.4, yc + 0.5), lw=1.4)
inputs(yc, "same orography\nfields", "input $x$  5.625°")
network(yc, True)
arrow((81.4, yc), (84.2, yc), lw=1.4)
img(D["pred_t2m"], 84.4, yc - 3.3, 13.0, 6.6, "RdYlBu_r", ec=RED)
T(91.0, yc - 6.2, "$\\hat y$  1.5°", fs=9, c=RED)
box(83.4, yc + 5.2, 15.0, 5.2, "", fc=RED, ec=RED)
T(90.9, yc + 7.8, "$\\|A\\hat y-x\\|\\approx 6\\times10^{-8}$", fs=9.2, c="white")
fig.savefig("figures/fig_overview.png"); plt.close(fig)
print("wrote")
