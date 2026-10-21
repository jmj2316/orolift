"""Architecture schematic in the style of downscaling papers: left-to-right pipeline with real
fields and feature maps, 3-D feature slabs, coarse/fine zones, and detail panels below."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle
import numpy as np

D = np.load("figures/arch_inputs.npz")
NAVY, BLUE, ORNG, GREEN, RED = "#2E4A6B", "#2C6FBB", "#E8873A", "#2E8B6E", "#B23A48"
GRAY, PALE = "#ECECEC", "#E3ECF7"
ZC, ZF = "#EAF1FA", "#FDF1E6"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
W, H = 100.0, 100.0
fig = plt.figure(figsize=(10.0, 10.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
FS, FSS, FST = 10.5, 9.2, 11.5


def shade(c, f):
    c = np.array(matplotlib.colors.to_rgb(c))
    return tuple(np.clip(c * f if f < 1 else c + (1 - c) * (f - 1), 0, 1))


def T(x, y, s, fs=FS, c="0.12", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=6, linespacing=1.25, **kw)


def box(x, y, w, h, t="", fc=GRAY, tc="0.12", fs=FS, r=0.8, ec="0.4", lw=0.9, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if t:
        T(x + w / 2, y + h / 2, t, fs=fs, c=tc)


def arrow(p, q, rad=0.0, c="0.25", lw=1.2, ls="-", ms=11, style="-|>", z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, color=c, lw=lw, ls=ls, mutation_scale=ms,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def slab(x, y, w, h, c, dep=0.9):
    """feature volume: width ~ channels, height ~ spatial size."""
    ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="0.2", lw=0.8, zorder=3))
    ax.add_patch(Polygon([(x, y + h), (x + dep, y + h + dep), (x + w + dep, y + h + dep), (x + w, y + h)],
                         fc=shade(c, 1.45), ec="0.2", lw=0.8, zorder=3))
    ax.add_patch(Polygon([(x + w, y), (x + w + dep, y + dep), (x + w + dep, y + h + dep), (x + w, y + h)],
                         fc=shade(c, 0.78), ec="0.2", lw=0.8, zorder=3))


def img(a, x, y, w, h, cmap, vmin=None, vmax=None, frames=0, fc="white", ec="0.3"):
    for k in range(frames, 0, -1):
        ax.add_patch(Rectangle((x + 0.75 * k, y + 0.75 * k), w, h, fc=fc, ec=ec, lw=0.8, zorder=2 + 0.01 * (5 - k)))
    ax.imshow(a, extent=[x, x + w, y, y + h], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest", zorder=3.5, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=0.9, zorder=4))


def panel(x, y, w, h, tag, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white",
                                ec="0.6", lw=0.9, ls=(0, (4, 3)), zorder=1))
    T(x + 1.3, y + h - 2.1, tag, fs=14, ha="left", fontweight="bold")
    T(x + 4.2, y + h - 2.1, title, fs=FST, ha="left", fontweight="bold")


# ═════════════════════════ (a) pipeline : row A (coarse grid)
T(1.2, 98.6, "a", fs=14, ha="left", fontweight="bold")
T(4.0, 98.6, "OroLIFT: trunk on the coarse grid, decoder queried on the fine grid", fs=12.5, ha="left", fontweight="bold")
def zone(x0, y0, x1, y1, fc):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec="0.7", lw=0.9, zorder=0))
zone(1, 65, 66.5, 96, ZC); zone(1, 33.5, 99, 63.5, ZF)
T(2.6, 93.8, "coarse grid  32 × 64  (2 048 pixels)", fs=FSS + 0.6, c=BLUE, ha="left", fontweight="bold")
T(21.0, 62.0, "fine grid  121 × 240  (29 040 pixels)", fs=FSS + 0.6, c="#B8631F", ha="left", fontweight="bold")

img(D["x_t2m"], 4.5, 81.5, 11.0, 5.5, "RdYlBu_r", frames=2)
T(10.0, 78.9, "ERA5 5.625° input\n69 × 32 × 64", fs=FSS)
img(D["oro_lr"], 4.5, 68.8, 11.0, 5.5, "terrain", frames=2, vmin=0, vmax=4500)
T(10.0, 66.6, "5 static fields\norography · mask · coords", fs=FSS - 0.5)
arrow((18.6, 85.6), (20.8, 78.4)); arrow((18.6, 72.8), (20.8, 75.6))
slab(21.5, 73.0, 2.2, 8.0, "#5B7FA6"); T(22.6, 69.8, "stem\n3×3 conv", fs=FSS)
arrow((25.0, 77.0), (26.7, 77.0))
for k in range(4):
    slab(27.5 + k * 2.6, 73.0, 1.9, 8.0, NAVY)
T(39.4, 77.0, "···", fs=14)
T(33.0, 70.6, "8 × (SAFM + CCM)", fs=FSS - 0.2)
T(33.0, 68.4, "1.6 M parameters", fs=FSS - 0.6, c=NAVY, style="italic")
arrow((41.0, 77.0), (42.4, 77.0))
slab(43.0, 73.0, 2.2, 8.0, "#5B7FA6"); T(44.1, 69.8, "tail\n3×3 conv", fs=FSS)
for xs, k in [(22.6, "stem"), (33.0, "mid"), (44.1, "tail")]:
    img(D[k][0], xs - 3.4, 84.0, 6.8, 3.4, "magma")
T(33.0, 90.6, "learned feature maps", fs=FSS - 0.4, c="0.35", style="italic")
arrow((46.9, 78.5), (49.0, 84.5), rad=-0.1); arrow((46.9, 75.5), (49.0, 71.5), rad=0.1)
slab(49.8, 83.0, 3.4, 8.0, GREEN); T(52.0, 94.4, "FiLM maps · 768 ch", fs=FSS - 0.2)
slab(49.8, 68.8, 1.3, 8.0, "#6BBF9D")
# lookup
box(59.8, 69.0, 5.8, 20.0, "", fc="white", ec=NAVY, z=4)
T(62.7, 79.0, "bilinear lookup", fs=FSS - 0.2, rotation=90, c=NAVY)
arrow((54.9, 87.0), (59.7, 87.0), lw=1.5, c=NAVY); arrow((52.5, 73.0), (59.7, 73.0), lw=1.5, c=NAVY)
T(54.6, 66.9, "direct latent · 32 ch", fs=FSS - 0.9)
ax.plot([62.7, 62.7], [69.0, 64.4], color=NAVY, lw=1.5, zorder=3)
ax.plot([62.7, 6.6], [64.4, 64.4], color=NAVY, lw=1.5, zorder=3)
arrow((6.6, 64.4), (6.6, 59.8), lw=1.5, c=NAVY, ms=12)
arrow((14.9, 64.4), (14.9, 59.8), lw=1.5, c=NAVY, ms=12)

# ═════════════════════════ row B (fine grid)
slab(6.0, 40.5, 1.3, 18.0, "#6BBF9D"); slab(8.8, 40.5, 1.7, 18.0, ORNG)
slab(12.0, 40.5, 0.8, 18.0, "#9A9A9A"); slab(14.4, 40.5, 3.4, 18.0, GREEN)
T(12.0, 37.2, "sampled latent · static HR\nδ, cell size · modulations", fs=FSS - 0.4)
cols = [(28.0, 4), (31.5, 5), (35.0, 5), (38.5, 3)]
pts = [[(cx, 49.0 + (i - (n - 1) / 2) * 3.2) for i in range(n)] for cx, n in cols]
for a_, b_ in zip(pts[:-1], pts[1:]):
    for p in a_:
        for q in b_:
            ax.plot([p[0], q[0]], [p[1], q[1]], color="0.65", lw=0.45, zorder=3)
for col in pts:
    for p in col:
        ax.add_patch(Circle(p, 1.05, fc=NAVY, ec="0.15", lw=0.7, zorder=4))
arrow((20.8, 49.0), (26.4, 49.0), lw=1.6, c=NAVY)
ax.plot([17.2, 17.2, 38.5], [59.5, 60.5, 60.5], color=GREEN, lw=1.2, zorder=3)
for cx, yb in [(31.5, 57.4), (35.0, 57.4), (38.5, 55.4)]:
    arrow((cx, 60.5), (cx, yb), c=GREEN, lw=1.2)
T(40.4, 60.5, "(γ, β) at every layer", fs=FSS - 0.4, c=GREEN, ha="left", style="italic")
T(33.5, 37.2, "implicit decoder (panel c)\nshared across pixels", fs=FSS - 0.2)
img(D["pred_t2m"] - D["up_t2m"], 47.0, 49.8, 11.0, 5.5, "RdBu_r", vmin=-4, vmax=4)
T(52.5, 47.7, "residual r ⊙ s", fs=FSS)
arrow((40.4, 49.0), (46.6, 52.4), lw=1.6, c=NAVY, rad=-0.05)
img(D["up_t2m"], 47.0, 40.5, 11.0, 5.5, "RdYlBu_r", vmin=D["truth_t2m"].min(), vmax=D["truth_t2m"].max())
T(52.5, 38.1, "Up(x)\nbilinear lookup of x", fs=FSS - 0.5)
ax.add_patch(Circle((63.5, 48.0), 1.3, fc="white", ec="0.2", lw=1.0, zorder=4)); T(63.5, 48.0, "+", fs=13)
arrow((58.2, 52.5), (62.4, 48.9), lw=1.3); arrow((58.2, 43.2), (62.4, 47.1), lw=1.3)
arrow((64.9, 48.0), (66.3, 48.0), lw=1.3)
box(66.4, 40.5, 4.6, 15.0, "", fc=RED, ec="0.2", r=0.8)
T(68.7, 48.0, "Π$_A$", fs=FS + 1.5, c="white", rotation=90)
T(68.7, 37.6, "projection\n(panel d)", fs=FSS - 0.2, c=RED)
vmin_, vmax_ = D["truth_t2m"].min(), D["truth_t2m"].max()
arrow((71.2, 48.0), (77.0, 52.0), lw=1.5, rad=-0.05)
img(D["pred_t2m"], 78.0, 48.8, 13.0, 6.5, "RdYlBu_r", vmin=vmin_, vmax=vmax_)
T(84.5, 57.6, "output ŷ  ·  69 × 121 × 240", fs=FSS)
img(D["truth_t2m"], 78.0, 37.8, 13.0, 6.5, "RdYlBu_r", vmin=vmin_, vmax=vmax_)
T(84.5, 35.6, "ERA5 1.5° truth", fs=FSS)
arrow((84.5, 45.0), (84.5, 48.2), style="<|-|>", lw=1.1, c="0.2", ms=8)
T(87.0, 46.6, "$\\mathcal{L}$", fs=FS + 1, c="0.2")
T(83.0, 46.6, "lat-weighted\nMSE", fs=FSS - 0.7, c="0.3", ha="right")

# ═════════════════════════ (b) SAFM block  (row A, right)
panel(68, 65, 31, 31, "b", "SAFM + CCM block (×8)")
for x0, s, lab in [(72.4, 5.0, "1"), (79.4, 3.7, "½"), (85.2, 2.6, "¼"), (90.0, 1.8, "⅛")]:
    ax.add_patch(Rectangle((x0, 84.5), s, s, fc="#C9D8EC", ec="0.3", lw=0.9, zorder=3))
    T(x0 + s / 2, 82.9, lab, fs=FSS)
T(83.5, 79.3, "4 channel groups, max-pooled to 1/2$^i$,\none depthwise 3×3 each", fs=FSS - 0.4)
arrow((83.5, 77.1), (83.5, 76.1))
box(70.0, 71.7, 27.0, 4.4, "upsample · concat · 1×1 → gate\n$x\\leftarrow x+\\mathrm{SAFM}(\\mathrm{LN}(x))$", fs=FSS - 0.5, fc=PALE)
arrow((83.5, 71.7), (83.5, 70.7))
box(70.0, 67.1, 27.0, 3.6, "CCM: 3×3 conv → 1×1\n$x\\leftarrow x+\\mathrm{CCM}(\\mathrm{LN}(x))$", fs=FSS - 0.5, fc=PALE)

# ═════════════════════════ row C
# (c) implicit decoder
panel(1, 1, 32, 31, "c", "Implicit decoder")
img(D["oro_detail"], 5.5, 20.6, 8.0, 3.8, "RdBu_r", vmin=-300, vmax=300)
T(9.5, 26.4, "sub-grid\norography", fs=FSS - 0.5)
img(D["direct"][0], 20.5, 20.6, 8.0, 3.8, "magma")
T(24.5, 26.4, "sampled\nlatent", fs=FSS - 0.5)
for x0, x1, lab, fc in [(3.0, 6.4, "δ", "#CFCFCF"), (6.4, 10.4, "cell", "#9A9A9A"),
                        (10.4, 17.4, "static HR", "#F2B27C"), (17.4, 31.0, "latent (32)", "#6BBF9D")]:
    ax.add_patch(Rectangle((x0, 17.0), x1 - x0, 2.8, fc=fc, ec="0.3", lw=0.8, zorder=3))
    T((x0 + x1) / 2, 18.4, lab, fs=FSS - 0.8)
arrow((9.5, 20.6), (13.5, 19.9), lw=0.9, ms=8); arrow((24.5, 20.6), (22.5, 19.9), lw=0.9, ms=8)
T(17.0, 15.2, "per-pixel query q  (43 channels)", fs=FSS - 0.4)
arrow((17, 14.2), (17, 13.2))
box(5.0, 10.2, 24.0, 3.0, "1×1 conv → 128", fs=FSS - 0.3, fc=PALE)
arrow((17, 10.2), (17, 9.4))
box(3.5, 5.0, 27.0, 4.4, "3 × GELU( W h · (1+γ) + β )\n(γ, β) from the sampled FiLM maps", fs=FSS - 0.7, fc="#D8EEE5")
arrow((17, 5.0), (17, 4.2))
box(5.0, 1.8, 24.0, 2.4, "1×1 conv → 69  (zero-init)", fs=FSS - 0.6, fc=PALE)

# (d) projection
panel(34.5, 1, 32, 31, "d", "Consistency projection")
box(36.5, 24.2, 28.0, 3.4, "$A=A_{lat}\\otimes A_{lon}$   (separable)", fs=FSS)
arrow((50.5, 24.2), (50.5, 23.0))
box(36.5, 12.6, 28.0, 10.4,
    "$R=X-A_{lat}\\,Y\\,A_{lon}^{\\top}$\n$S=M_{lat}^{-1}\\,R\\,M_{lon}^{-1}$\n"
    "$Y^{\\star}=Y+D_{lat}^{-1}A_{lat}^{\\top}\\,S\\,A_{lon}D_{lon}^{-1}$", fs=FSS - 0.2, fc=PALE)
T(50.5, 10.4, "$M=AD^{-1}A^{\\top}$ is 32×32 and 64×64\nzero parameters · differentiable", fs=FSS - 0.6, c="0.3")
arrow((50.5, 8.6), (50.5, 7.8))
box(36.5, 2.2, 28.0, 5.4, "$\\|Ay-x\\|$ = 6×10$^{-8}$ (fp32), 7×10$^{-13}$ (fp64)\n(unconstrained: 9×10$^{-2}$)", fs=FSS - 0.7, fc=RED, tc="white")

# (e) probabilistic heads
panel(68, 1, 31, 31, "e", "Probabilistic heads")
box(70.0, 23.4, 12.0, 3.2, "frozen  $y_{det}$", fs=FSS - 0.2, fc=NAVY, tc="white")
box(85.0, 23.4, 12.0, 3.2, "truth  $y$", fs=FSS - 0.2, fc=BLUE, tc="white")
arrow((76.0, 23.4), (80.5, 21.2)); arrow((91.0, 23.4), (86.5, 21.2))
box(70.0, 17.4, 27.0, 3.6, "$e=y-y_{det}\\in\\ker A$", fs=FSS, fc=PALE)
arrow((77.0, 17.4), (76.5, 14.6)); arrow((90.0, 17.4), (90.5, 14.6))
box(69.2, 2.0, 13.4, 12.4, "variance head\nGaussian NLL\n$\\log\\sigma(x)$\nper-pixel\nspread", fs=FSS - 0.5, fc=GREEN, tc="white")
box(84.4, 2.0, 13.4, 12.4, "flow head\nvelocity in ker A\n→ every member\nsatisfies\n$Ay=x$", fs=FSS - 0.5, fc=GREEN, tc="white")

ax.set_xlim(0, W); ax.set_ylim(0, H)
fig.savefig("figures/fig0_architecture.png", dpi=200, facecolor="white")
print("ok")
