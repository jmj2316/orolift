"""Two independent estimates of the same quantity.

The flow head measures Var(y|x) as ensemble spread; the heteroscedastic head fits
it directly under a Gaussian likelihood.  They share only the frozen mean model, so
agreement is evidence that f_irr is a property of the problem rather than of one
generative model -- and disagreement localizes which of the two is miscalibrated.
"""
import os, sys, json, importlib.util
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, area_weights
from orolift.models.hetero import VarHead
from orolift.varlist import VARS_69, LEVELS, ATMOS_PREFIXES
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
N = 240; BS = 8


def main():
    z = np.load("runs/irreducible.npz")
    num_f, den = z["num"], z["den"]                 # flow: E[Var(y|x)], E[(y-base)^2]

    mean, std = load_norm()
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs_np = np.load(f"{DATA}/resid_std.npy")
    rs = torch.tensor(rs_np, device=dev)[None, :, None, None]
    det, _ = ev.load_model("runs/orolift_base/best.pt", rs_np, dev); det.eval()
    dh = torch.load("runs/hetero_base/best.pt", map_location="cpu", weights_only=False)
    het = VarHead(dim=dh["cfg"]["dim"], depth=dh["cfg"]["depth"]).to(dev)
    ph = dict(het.named_parameters())
    with torch.no_grad():
        for k, v in dh["ema"].items():
            ph[k].copy_(v.to(dev))
    het.eval()
    print(f"hetero step {dh['step']}")

    num_h = torch.zeros(len(VARS_69), len(LAT_HR), len(LON_HR), device=dev, dtype=torch.float64)
    n = 0
    dl = torch.utils.data.DataLoader(HR15("test", limit=N), batch_size=BS, num_workers=4)
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            yd = det(x, sl, sh).float()
            s = torch.exp(het(yd / rs, sh).float())          # sigma in residual units
        num_h += ((s * rs * st) ** 2).sum(0).double()
        n += y.shape[0]
    num_h = (num_h / n).cpu().numpy()

    W = np.broadcast_to(area_weights()[:, None], den.shape[1:])
    gf = np.array([(num_f[i] * W).sum() / (den[i] * W).sum() for i in range(len(VARS_69))])
    gh = np.array([(num_h[i] * W).sum() / (den[i] * W).sum() for i in range(len(VARS_69))])

    print(f"\n{'variable':26s}{'f_irr flow':>12s}{'f_irr hetero':>14s}{'ratio':>8s}")
    for v in ["2m_temperature", "mean_sea_level_pressure", "geopotential_500",
              "temperature_850", "u_component_of_wind_850", "specific_humidity_700",
              "specific_humidity_925", "u_component_of_wind_250"]:
        i = VARS_69.index(v)
        print(f"{v:26s}{gf[i]:12.3f}{gh[i]:14.3f}{gf[i]/max(gh[i],1e-9):8.3f}")
    print(f"\nacross all 69 variables:  corr {np.corrcoef(gf, gh)[0,1]:.4f}   "
          f"median ratio {np.median(gf/np.maximum(gh,1e-9)):.3f}   "
          f"mean |log ratio| {np.mean(np.abs(np.log(gf/np.maximum(gh,1e-9)))):.3f}")
    sp = []
    for i in range(len(VARS_69)):
        a_, b_ = num_f[i].ravel(), num_h[i].ravel()
        sp.append(np.corrcoef(a_, b_)[0, 1])
    print(f"spatial correlation of the two variance maps: median {np.median(sp):.3f} "
          f"(10-90% {np.percentile(sp,10):.2f}-{np.percentile(sp,90):.2f})")
    json.dump({"f_flow": gf.tolist(), "f_het": gh.tolist(), "spatial_corr": sp,
               "vars": VARS_69}, open("runs/eval/sigma_compare.json", "w"))

    # ---------------- figure
    fig = plt.figure(figsize=(12.5, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    col = {"geopotential": "C0", "temperature": "C3", "u_component_of_wind": "C1",
           "v_component_of_wind": "C2", "specific_humidity": "C4"}
    for pfx, c in col.items():
        idx = [VARS_69.index(f"{pfx}_{l}") for l in LEVELS]
        ax.scatter(gh[idx], gf[idx], s=16, c=c, label=pfx.replace("_component_of_wind", " wind"))
    idx = [VARS_69.index(v) for v in
           ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure"]]
    ax.scatter(gh[idx], gf[idx], s=26, c="k", marker="*", label="surface")
    lim = [0, max(gf.max(), gh.max()) * 1.05]
    ax.plot(lim, lim, "k-", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$f_{irr}$  heteroscedastic"); ax.set_ylabel(r"$f_{irr}$  flow ensemble")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title(f"(a) 69 variables, corr = {np.corrcoef(gf, gh)[0,1]:.3f}", fontsize=9)

    axp = fig.add_subplot(gs[0, 1])
    for pfx, c in col.items():
        idx = [VARS_69.index(f"{pfx}_{l}") for l in LEVELS]
        axp.plot(gf[idx], LEVELS, "-o", ms=3, color=c, lw=1.1)
        axp.plot(gh[idx], LEVELS, "--s", ms=3, color=c, lw=1.0, alpha=0.6)
    axp.invert_yaxis(); axp.set_yscale("log")
    axp.set_yticks(LEVELS, [str(l) for l in LEVELS], fontsize=6)
    axp.set_xlabel(r"$f_{irr}$"); axp.set_ylabel("pressure (hPa)")
    axp.set_title("(b) solid = flow, dashed = heteroscedastic", fontsize=9)

    axs = fig.add_subplot(gs[0, 2])
    axs.hist(sp, bins=25, color="C0")
    axs.set_xlabel("spatial correlation of the two variance maps")
    axs.set_title("(c) per variable", fontsize=9); axs.set_yticks([])

    lo, la = np.meshgrid(LON_HR, LAT_HR)
    lsm = sh[0, 3].cpu().numpy()
    cmap = plt.get_cmap("inferno").copy(); cmap.set_bad("0.75")
    v = "u_component_of_wind_850"; i = VARS_69.index(v)
    thr = 0.15 * float((den[i] * W).sum() / W.sum())
    msk = den[i] >= thr
    for k, (src, lab) in enumerate([(num_f, "flow ensemble"), (num_h, "heteroscedastic")]):
        A = fig.add_subplot(gs[1, k])
        f = np.where(msk, np.clip(src[i] / np.maximum(den[i], 1e-30), 0, 1), np.nan)
        im = A.pcolormesh(lo, la, f, cmap=cmap, vmin=0, vmax=1, shading="auto")
        A.contour(lo, la, lsm, levels=[0.5], colors="w", linewidths=0.4)
        A.set_xticks([]); A.set_yticks([]); A.set_title(f"(d) {v} — {lab}", fontsize=9)
    A = fig.add_subplot(gs[1, 2])
    d_ = np.where(msk, np.clip(num_f[i] / np.maximum(den[i], 1e-30), 0, 1)
                  - np.clip(num_h[i] / np.maximum(den[i], 1e-30), 0, 1), np.nan)
    imd = A.pcolormesh(lo, la, d_, cmap="RdBu_r", vmin=-0.4, vmax=0.4, shading="auto")
    A.contour(lo, la, lsm, levels=[0.5], colors="k", linewidths=0.4)
    A.set_xticks([]); A.set_yticks([]); A.set_title("(e) flow − heteroscedastic", fontsize=9)
    fig.colorbar(imd, ax=A, shrink=0.8)
    fig.suptitle("Do two independent estimators agree on how much is irreducible?", y=1.04)
    fig.savefig("figures/fig15_sigma_compare.png", bbox_inches="tight")
    print("\nwrote figures/fig15_sigma_compare.png")


if __name__ == "__main__":
    main()
