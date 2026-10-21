"""Fig. 18 (restyled): where the generative head places its uncertainty, judged against realized error."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import tag, note, ACCENT

RUNS = [("base", "0.70 M", "#B7C9E8"), ("nocond", "1.81 M", "#3F6BBE"), ("v2", "1.81 M + $\\sigma$ conditioning", ACCENT)]
KEY = [("2m_temperature", "2 m T"), ("mean_sea_level_pressure", "MSLP"), ("geopotential_500", "Z500"),
       ("temperature_850", "T850"), ("u_component_of_wind_850", "u850"),
       ("specific_humidity_700", "q700"), ("specific_humidity_925", "q925")]

R = {t: json.load(open(f"runs/eval/sigma_referee_{t}.json")) for t, _, _ in RUNS}
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.5), gridspec_kw=dict(width_ratios=[1.75, 1], wspace=0.22))

a = ax[0]
w = 0.26
x = np.arange(len(KEY))
for j, (t, lab, c) in enumerate(RUNS):
    v = [R[t]["rows"][k][1] for k, _ in KEY]
    a.bar(x + (j - 1) * w, v, w * 0.92, color=c, label=lab, zorder=3)
het = [R["v2"]["rows"][k][2] for k, _ in KEY]
for i, h in enumerate(het):
    a.plot([i - 1.5 * w, i + 1.5 * w], [h, h], color="k", lw=1.6, zorder=4, solid_capstyle="butt",
           label="independent variance head" if i == 0 else None)
a.set_xticks(x, [l for _, l in KEY]); a.set_ylim(0, 1.27); a.set_yticks(np.arange(0, 1.01, 0.2))
a.set_ylabel("Spatial correlation with realized error")
a.legend(ncol=2, loc="upper left", fontsize=8, columnspacing=1.2, handlelength=1.4)
a.set_title("Flow-head spread against realized deterministic error", fontsize=9.5, loc="left")
tag(a, "a", dx=-0.07)

b = ax[1]
gain = np.array([R["v2"]["rows"][k][1] - R["nocond"]["rows"][k][1] for k, _ in KEY])
cols = [ACCENT if lab in ("q700", "q925") else "0.65" for _, lab in KEY]
b.bar(x, gain, 0.62, color=cols, zorder=3)
for i, g in enumerate(gain):
    b.text(i, g + (0.008 if g >= 0 else -0.008), f"{g:+.2f}", ha="center", va="bottom" if g >= 0 else "top", fontsize=7.5)
b.axhline(0, color="k", lw=0.7)
b.set_xticks(x, [l for _, l in KEY]); b.set_ylim(min(-0.04, gain.min() - 0.04), gain.max() + 0.07)
b.set_ylabel("Gain from $\\sigma$ conditioning (correlation)")
b.set_title("Conditioning helps moisture", fontsize=9.5, loc="left")
tag(b, "b", dx=-0.17)
for t_ in (a, b):
    t_.tick_params(axis="x", labelsize=8)
    t_.spines[["top", "right"]].set_visible(False)

med = {t: R[t]["median"][1] for t, _, _ in RUNS}
fig.savefig("figures/fig18_flowabl_new.png")
print({k: round(v, 3) for k, v in med.items()}, "het median", round(R["v2"]["median"][2], 3))
