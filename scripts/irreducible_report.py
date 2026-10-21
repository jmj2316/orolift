"""Re-analysis of runs/irreducible.npz with the right estimator.

f_irr must be a ratio of area-weighted *means*, not a mean of pointwise ratios:
the pointwise ratio blows up wherever the denominator (the sub-grid signal itself)
is small, so averaging it measures the quiet points, not the atmosphere.
"""
import numpy as np, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.grids import LAT_HR, LON_HR, area_weights
from orolift.varlist import VARS_69, LEVELS
from orolift.data import static_features

# The referee test (scripts/sigma_referee.py) showed the heteroscedastic head tracks
# the realized error far better than the flow ensemble does (spatial corr 0.86 vs 0.31),
# so the *maps* come from it.  The flow head keeps its own job: spectra and exact
# ker(A) sampling.
SRC = sys.argv[1] if len(sys.argv) > 1 else "hetero"
if SRC == "hetero":
    z = np.load("runs/sigma_referee.npz"); num, den = z["het"], z["den"]
else:
    z = np.load("runs/irreducible.npz"); num, den = z["num"], z["den"]
print(f"source: {SRC}\n")
W = np.broadcast_to(area_weights()[:, None], num.shape[1:])

def ratio(i, m=None):
    ww = W if m is None else W * m
    return float((num[i] * ww).sum() / (den[i] * ww).sum())

MAPS = ["2m_temperature", "geopotential_500", "u_component_of_wind_850", "specific_humidity_700"]
trop = np.broadcast_to((np.abs(LAT_HR) <= 20)[:, None], num.shape[1:]).astype(float)
ext = np.broadcast_to((np.abs(LAT_HR) > 30)[:, None], num.shape[1:]).astype(float)
sh, _ = static_features("cpu"); lsm = sh[3].numpy(); oro = sh[0].numpy() * 1000
land = np.broadcast_to(lsm > 0.5, num.shape[1:]).astype(float)
sea = 1 - land
mount = (np.abs(sh[1].numpy() * 1000) > 300).astype(float)

print(f"{'variable':26s}{'global':>9s}{'tropics':>9s}{'extratrop':>11s}{'land':>8s}{'ocean':>8s}{'|oro|>300m':>12s}")
for v in MAPS + ["temperature_850", "specific_humidity_925", "geopotential_850",
                 "u_component_of_wind_250", "10m_u_component_of_wind"]:
    i = VARS_69.index(v)
    print(f"{v:26s}{ratio(i):9.3f}{ratio(i,trop):9.3f}{ratio(i,ext):11.3f}"
          f"{ratio(i,land):8.3f}{ratio(i,sea):8.3f}{ratio(i,mount):12.3f}")

lo, la = np.meshgrid(LON_HR, LAT_HR)

# A fraction is meaningless where its denominator is noise: over the ocean there is
# essentially no sub-grid 2 m temperature signal at all, so "100% irreducible" there
# says nothing about the atmosphere.  Mask, and show the signal itself underneath.
def masked_frac(i, rel=0.15):
    d = den[i]
    thr = rel * float((d * W).sum() / W.sum())
    f = np.where(d >= thr, num[i] / np.maximum(d, 1e-30), np.nan)
    return np.clip(f, 0, 1), np.sqrt(d), float((d >= thr).mean())

UNITS = {"2m_temperature": ("K", 1), "geopotential_500": (r"m$^2$s$^{-2}$", 1),
         "u_component_of_wind_850": (r"m s$^{-1}$", 1),
         "specific_humidity_700": (r"g kg$^{-1}$", 1e3)}
cmap = plt.get_cmap("inferno").copy(); cmap.set_bad("0.75")
fig, ax = plt.subplots(2, 4, figsize=(14, 5.4), constrained_layout=True)
for k, v in enumerate(MAPS):
    i = VARS_69.index(v)
    f, sig, cov = masked_frac(i)
    im = ax[0, k].pcolormesh(lo, la, f, cmap=cmap, vmin=0, vmax=1, shading="auto")
    ax[0, k].contour(lo, la, lsm, levels=[0.5], colors="w", linewidths=0.4)
    ax[0, k].set_title(f"{v}\nglobal $f_{{irr}}$ = {ratio(i):.2f}   ({100*cov:.0f}% of area shown)",
                       fontsize=8.5)
    u, sc = UNITS[v]
    im2 = ax[1, k].pcolormesh(lo, la, sig * sc, cmap="viridis", shading="auto",
                              vmax=np.percentile(sig * sc, 99))
    ax[1, k].contour(lo, la, lsm, levels=[0.5], colors="w", linewidths=0.4)
    ax[1, k].set_title(f"total sub-grid signal ({u})", fontsize=8)
    fig.colorbar(im2, ax=ax[1, k], shrink=0.8, pad=0.01)
    for A in (ax[0, k], ax[1, k]):
        A.set_xticks([]); A.set_yticks([])
fig.colorbar(im, ax=ax[0, :], shrink=0.8, pad=0.01,
             label="irreducible fraction (gray: no sub-grid signal to speak of)")
fig.suptitle("How much of the missing fine structure is beyond any model?"
             "   1 = the coarse state says nothing about it", y=1.05)
fig.savefig(f"figures/fig13_irreducible_maps.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
for k, pfx in enumerate(["geopotential", "temperature", "u_component_of_wind", "specific_humidity"]):
    cs = np.zeros((len(LEVELS), len(LAT_HR))); dz = np.zeros_like(cs)
    for a_, l in enumerate(LEVELS):
        i = VARS_69.index(f"{pfx}_{l}")
        dz[a_] = (den[i] * W).sum(1)
        cs[a_] = (num[i] * W).sum(1) / np.maximum(dz[a_], 1e-30)
    cs = np.where(dz >= 0.15 * dz.mean(1, keepdims=True), np.clip(cs, 0, 1), np.nan)
    im = ax[k].pcolormesh(LAT_HR, np.arange(len(LEVELS)), cs, cmap=cmap,
                          vmin=0, vmax=1, shading="auto")
    ax[k].set_yticks(range(len(LEVELS)), [str(l) for l in LEVELS], fontsize=6)
    ax[k].invert_yaxis(); ax[k].set_xlabel("latitude")
    ax[k].set_title(pfx.replace("_component_of_wind", " wind"), fontsize=9)
    for x0 in (-20, 20):
        ax[k].axvline(x0, color="c", lw=0.8, ls=":")
    if k == 0: ax[k].set_ylabel("pressure (hPa)")
fig.colorbar(im, ax=ax, shrink=0.85, label="irreducible fraction")
fig.suptitle("Zonal-mean irreducible fraction — the tropics keep their secrets, "
             "the extratropics do not", y=1.09)
fig.savefig("figures/fig14_irreducible_section.png", bbox_inches="tight")
print("\nwrote figures/fig13_irreducible_maps.png, figures/fig14_irreducible_section.png")
