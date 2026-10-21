"""Where is sub-grid structure irreducible?

Raw ensemble spread is not a map of irreducibility -- it is mostly a map of where
the variable is large.  The interpretable quantity is the *fraction* of the sub-grid
signal that no model can recover:

    f_irr(loc) = E[ Var(y | x) ] / E[ (y - baseline)^2 ]

numerator from the ker(A)-constrained ensemble, denominator from the free
(bilinear + projection) baseline residual.  f_irr -> 1 means the coarse state says
nothing about the fine structure there; f_irr -> 0 means the fine structure is
slaved to the resolved scales.

Var(y|x) is estimated around *this* deterministic mean, so f_irr is an upper bound on
the irreducible fraction.
"""
import argparse, os, sys, json, importlib.util
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, area_weights
from orolift.models.flow import FlowHead, FlowSampler
from orolift.varlist import VARS_69, LEVELS, ATMOS_PREFIXES
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
MAPS = ["2m_temperature", "geopotential_500", "u_component_of_wind_850", "specific_humidity_700"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/flow_base/best.pt")
    ap.add_argument("--det", default="runs/orolift_base/best.pt")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--members", type=int, default=12)
    ap.add_argument("--ode_steps", type=int, default=12)
    a = ap.parse_args()

    mean, std = load_norm()
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs_np = np.load(f"{DATA}/resid_std.npy")
    det, _ = ev.load_model(a.det, rs_np, dev); det.eval()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = FlowHead(dim=d["cfg"]["dim"], depth=d["cfg"]["depth"], resid_std=rs_np).to(dev)
    pd = dict(net.named_parameters())
    with torch.no_grad():
        for k, v in d["ema"].items():
            pd[k].copy_(v.to(dev))
    net.eval()
    smp = FlowSampler(net, C, det)

    V, H, W = len(VARS_69), len(LAT_HR), len(LON_HR)
    num = torch.zeros(V, H, W, device=dev, dtype=torch.float64)   # E[Var(y|x)]
    den = torch.zeros(V, H, W, device=dev, dtype=torch.float64)   # E[(y-baseline)^2]
    rec = torch.zeros(V, H, W, device=dev, dtype=torch.float64)   # E[(y_det-baseline)^2]
    n = 0
    dl = torch.utils.data.DataLoader(HR15("test", limit=a.n), batch_size=a.bs, num_workers=4)
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        ens = smp.sample(x, sl, sh, n_members=a.members, steps=a.ode_steps)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            yd = det(x, sl, sh).float()
        base = C.project(C.upsample(x), x)
        num += (ens.var(1, unbiased=True) * st ** 2).sum(0).double()
        den += (((y - base) * st) ** 2).sum(0).double()
        rec += (((yd - base) * st) ** 2).sum(0).double()
        n += y.shape[0]
    num, den, rec = num / n, den / n, rec / n
    frac = (num / den.clamp_min(1e-30)).clamp(0, 1.5).cpu().numpy()
    np.savez("runs/irreducible.npz", frac=frac, num=num.cpu().numpy(),
             den=den.cpu().numpy(), rec=rec.cpu().numpy(), vars=np.array(VARS_69))

    w = area_weights()[:, None]
    gm = lambda f: float((f * w).mean() / w.mean())
    print(f"{'variable':26s}{'global f_irr':>14s}{'tropics 20S-20N':>18s}{'extratropics':>14s}")
    trop = (np.abs(LAT_HR) <= 20)[:, None]; ext = (np.abs(LAT_HR) > 30)[:, None]
    for v in MAPS + ["temperature_850", "specific_humidity_925", "geopotential_850"]:
        i = VARS_69.index(v)
        f = frac[i]
        print(f"{v:26s}{gm(f):14.3f}"
              f"{float((f*w*trop).sum()/(w*trop).sum()):18.3f}"
              f"{float((f*w*ext).sum()/(w*ext).sum()):14.3f}")

    # ---------------- maps
    lo, la = np.meshgrid(LON_HR, LAT_HR)
    lsm = sh[0, 3].cpu().numpy()
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 5.6), constrained_layout=True)
    for k, v in enumerate(MAPS):
        A = ax.flat[k]
        im = A.pcolormesh(lo, la, frac[VARS_69.index(v)], cmap="inferno", vmin=0, vmax=1,
                          shading="auto")
        A.contour(lo, la, lsm, levels=[0.5], colors="w", linewidths=0.4)
        A.set_xticks([]); A.set_yticks([])
        A.set_title(f"{v}   global mean {gm(frac[VARS_69.index(v)]):.2f}", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="irreducible fraction of sub-grid variance")
    fig.suptitle("How much of the missing fine structure is beyond any model?", y=1.04)
    fig.savefig("figures/fig13_irreducible_maps.png", bbox_inches="tight"); plt.close(fig)

    # ---------------- latitude-pressure cross sections
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    for k, pfx in enumerate(["geopotential", "temperature", "u_component_of_wind",
                             "specific_humidity"]):
        cs = np.array([[float((frac[VARS_69.index(f"{pfx}_{l}")][j] * 1).mean())
                        for l in LEVELS] for j in range(H)]).T
        im = ax[k].pcolormesh(LAT_HR, np.arange(len(LEVELS)), cs, cmap="inferno",
                              vmin=0, vmax=1, shading="auto")
        ax[k].set_yticks(range(len(LEVELS)), [str(l) for l in LEVELS], fontsize=6)
        ax[k].invert_yaxis(); ax[k].set_xlabel("latitude")
        ax[k].set_title(pfx.replace("_component_of_wind", " wind"), fontsize=9)
        if k == 0:
            ax[k].set_ylabel("pressure (hPa)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="irreducible fraction")
    fig.suptitle("Zonal-mean irreducible fraction — the atmosphere's unrecoverable scales", y=1.08)
    fig.savefig("figures/fig14_irreducible_section.png", bbox_inches="tight")
    print("\nwrote figures/fig13_irreducible_maps.png, figures/fig14_irreducible_section.png")


if __name__ == "__main__":
    main()
