"""How does the model reconstruct a cyclone core that the input barely resolves?

Two probes on the Haishen case:
  (a) radial MSLP profile -- the consistency constraint pins the *area average* over
      each 5.625 deg cell, so deepening the eye is only allowed if the surroundings
      rise.  If that is what the model does, the mechanism is "redistribute a fixed
      integral according to a learned shape prior", not "invent pressure".
  (b) input-gradient saliency of the reconstructed central pressure -- which of the
      69 input channels, and how far away, the answer actually depends on.
"""
import os, sys, importlib.util, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, LAT_LR, LON_LR
from orolift.varlist import VARS_69, LEVELS, ATMOS_PREFIXES
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
TIME = "2020-09-05 12:00:00"
BOX = (105, 175, 5, 40)


def main():
    mean, std = load_norm()
    mt = torch.tensor(mean, device=dev)[:, None, None]
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    model, _ = ev.load_model("runs/orolift_base/best.pt", np.load(f"{DATA}/resid_std.npy"), dev)
    ip = VARS_69.index("mean_sea_level_pressure")

    ds = HR15("test")
    ti = int(np.where(np.array([str(t) for t in ds.times]) == TIME)[0][0])
    y = ds[ti][None].to(dev); x = C.coarsen(y)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        p = model(x, sl, sh).float()
    b = C.project(C.upsample(x), x)
    hPa = lambda a: (a[0, ip] * st[ip] + mt[ip]).cpu().numpy() / 100

    msk = ((LAT_HR >= BOX[2]) & (LAT_HR <= BOX[3]))[:, None] & ((LON_HR >= BOX[0]) & (LON_HR <= BOX[1]))[None, :]
    Ptr = hPa(y); j, i = np.unravel_index(np.argmin(np.where(msk, Ptr, 1e9)), Ptr.shape)
    clat, clon = LAT_HR[j], LON_HR[i]
    print(f"center at {clat:.1f}N {clon:.1f}E  (truth {Ptr[j,i]:.1f} hPa)")

    # ---------------- (a) radial profile
    dy = (LAT_HR[:, None] - clat) * 111.0
    dx = (LON_HR[None, :] - clon) * 111.0 * np.cos(np.deg2rad(LAT_HR[:, None]))
    r = np.sqrt(dx ** 2 + dy ** 2)
    edges = np.arange(0, 1300, 100)
    prof = {}
    for k, F in [("ERA5 1.5° truth", Ptr), ("bilinear + projection", hPa(b)),
                 ("OroLIFT", hPa(p)), ("5.625° input (upsampled)", hPa(C.upsample(x)))]:
        prof[k] = [F[(r >= a) & (r < c)].mean() for a, c in zip(edges[:-1], edges[1:])]
    rc = 0.5 * (edges[:-1] + edges[1:])

    # the 5.625 cell mean must be identical for every consistent field
    ci = int(np.argmin(np.abs(LAT_LR - clat))); cj = int(np.argmin(np.abs(LON_LR - clon)))
    cell = lambda a: float((C.coarsen(a)[0, ip] * st[ip] + mt[ip])[ci, cj]) / 100
    print(f"5.625° cell mean containing the eye:  input {float((x[0,ip]*st[ip]+mt[ip])[ci,cj])/100:.2f}  "
          f"baseline {cell(b):.2f}  OroLIFT {cell(p):.2f}  truth {cell(y):.2f} hPa")

    # ---------------- (b) saliency
    x2 = x.clone().requires_grad_(True)
    out = model(x2, sl, sh)
    out[0, ip, j, i].backward()
    g = x2.grad[0].abs().detach()                       # (69, 32, 64)
    per_var = g.sum(dim=(1, 2)).cpu().numpy()
    per_var = per_var / per_var.sum()
    order = np.argsort(-per_var)
    print("\ntop input channels for the reconstructed central pressure:")
    for k in order[:10]:
        print(f"   {VARS_69[k]:28s} {100*per_var[k]:5.1f}%")
    grp = {}
    for k, v in enumerate(VARS_69):
        gname = next((pfx for pfx in ATMOS_PREFIXES if v.startswith(pfx + "_")), "surface")
        grp[gname] = grp.get(gname, 0) + per_var[k]
    print("  by group: " + "  ".join(f"{k}={100*v:.1f}%" for k, v in
                                     sorted(grp.items(), key=lambda t: -t[1])))
    gs = g.sum(0).cpu().numpy()
    rr = np.sqrt(((LAT_LR[:, None] - clat) * 111.0) ** 2 +
                 ((LON_LR[None, :] - clon) * 111.0 * np.cos(np.deg2rad(LAT_LR[:, None]))) ** 2)
    tot = gs.sum()
    for R in [400, 800, 1500, 3000]:
        print(f"  fraction of sensitivity within {R:5d} km of the eye: {gs[rr<R].sum()/tot:.3f}")

    # ---------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2), constrained_layout=True)
    for k, (c, ls) in zip(prof, [("k", "-"), ("C2", "-."), ("C0", "-"), ("C1", "--")]):
        ax[0].plot(rc, prof[k], color=c, ls=ls, lw=1.4, label=k)
    ax[0].set_xlabel("radius from eye (km)"); ax[0].set_ylabel("azimuthal mean MSLP (hPa)")
    ax[0].legend(frameon=False, fontsize=6.5); ax[0].set_title("(a) radial profile")
    ax[0].axhline(prof["ERA5 1.5° truth"][-1], color="gray", lw=0.5, ls=":")

    d = np.array(prof["OroLIFT"]) - np.array(prof["bilinear + projection"])
    ax[1].bar(rc, d, width=90, color=np.where(d < 0, "C0", "C3"))
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("radius from eye (km)"); ax[1].set_ylabel("OroLIFT − baseline (hPa)")
    ax[1].set_title("(b) what the model moved\n(blue = deepened, red = raised)")

    sal = per_var[[VARS_69.index(f"{pf}_{l}") for pf in ATMOS_PREFIXES for l in LEVELS]]
    sal = sal.reshape(5, 13)
    im = ax[2].imshow(100 * sal, cmap="magma_r", aspect="auto")
    ax[2].set_xticks(range(13), [str(l) for l in LEVELS], fontsize=6, rotation=90)
    ax[2].set_yticks(range(5), ["Z", "u", "v", "T", "q"], fontsize=7)
    ax[2].set_title("(c) input sensitivity of the\nreconstructed eye pressure (%)")
    fig.colorbar(im, ax=ax[2], shrink=0.9)
    fig.suptitle(f"Why the cyclone comes back — {TIME}", y=1.06)
    fig.savefig("figures/fig8_typhoon_why.png", bbox_inches="tight")
    print("\nwrote figures/fig8_typhoon_why.png")


if __name__ == "__main__":
    main()
