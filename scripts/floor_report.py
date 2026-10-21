"""Step 4: is f_irr a floor, or just our model's deficiency?

For each deterministic mean model along a capacity axis (dim 48/96/192) and a data
axis (1/8, 1/2, full), an independent variance head is fitted and f_irr recomputed.
If the mean model keeps improving while f_irr flattens, f_irr is a property of the
problem.  If f_irr tracks the mean model down, it is just unexplained-by-us.
"""
import os, sys, json, importlib.util
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, area_weights, LAT_HR, LON_HR
from orolift.metrics import lat_weighted_rmse
from orolift.models.hetero import VarHead
from orolift.varlist import VARS_69
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
# (run, axis, x-value, label)
POINTS = [("sc_dim48",     "capacity", 48,     "dim 48"),
          ("cmp_orolift",  "capacity", 96,     "dim 96"),
          ("sc_dim192",    "capacity", 192,    "dim 192"),
          ("sc_data8",     "data",     0.125,  "1/8 data"),
          ("sc_data2",     "data",     0.5,    "1/2 data"),
          ("cmp_orolift",  "data",     1.0,    "full data"),
          ("orolift_base", "steps",    50000,  "dim 96, 50k")]
KEY = ["2m_temperature", "mean_sea_level_pressure", "geopotential_500",
       "temperature_850", "u_component_of_wind_850", "specific_humidity_700"]
N, BS = 192, 8


def evaluate_pair(run):
    mean, std = load_norm()
    mt = torch.tensor(mean, device=dev)[:, None, None]
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs_np = np.load(f"{DATA}/resid_std.npy")
    rs = torch.tensor(rs_np, device=dev)[None, :, None, None]
    det, _ = ev.load_model(f"runs/{run}/best.pt", rs_np, dev); det.eval()
    dh = torch.load(f"runs/het_{run}/best.pt", map_location="cpu", weights_only=False)
    het = VarHead(dim=dh["cfg"]["dim"], depth=dh["cfg"]["depth"]).to(dev)
    ph = dict(het.named_parameters())
    with torch.no_grad():
        for k, v in dh["ema"].items():
            ph[k].copy_(v.to(dev))
    het.eval()

    V, H, W_ = len(VARS_69), len(LAT_HR), len(LON_HR)
    num = torch.zeros(V, H, W_, device=dev, dtype=torch.float64)
    den = torch.zeros_like(num); rmse = 0; n = 0
    dl = torch.utils.data.DataLoader(HR15("test", limit=N), batch_size=BS, num_workers=4)
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            yd = det(x, sl, sh).float()
            s = torch.exp(het(yd / rs, sh).float())
        base = C.project(C.upsample(x), x)
        num += ((s * rs * st) ** 2).sum(0).double()
        den += (((y - base) * st) ** 2).sum(0).double()
        rmse = rmse + lat_weighted_rmse(yd * st + mt, y * st + mt) * y.shape[0]
        n += y.shape[0]
    Wt = torch.as_tensor(np.broadcast_to(area_weights()[:, None], (H, W_)).copy(), device=dev)
    f = ((num * Wt).sum(dim=(-2, -1)) / (den * Wt).sum(dim=(-2, -1))).cpu().numpy()
    return f, (rmse / n).cpu().numpy()


def main():
    out = {}
    for run, axis, xv, lab in POINTS:
        if not (os.path.exists(f"runs/{run}/best.pt") and os.path.exists(f"runs/het_{run}/best.pt")):
            print(f"skip {run} (missing)"); continue
        if run in out:
            continue
        f, r = evaluate_pair(run)
        out[run] = dict(f=f.tolist(), rmse=r.tolist())
        print(f"{lab:14s} f_irr " + " ".join(f"{v[:5]}={f[VARS_69.index(v)]:.3f}" for v in KEY)
              + "  |  det RMSE " + " ".join(f"{r[VARS_69.index(v)]:.4g}" for v in KEY), flush=True)
    json.dump(out, open("runs/floor.json", "w"))

    b = json.load(open("runs/eval/baseline_test_synthetic.json"))
    br = np.array(b["rmse"])
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 6.2), constrained_layout=True)
    for k, v in enumerate(KEY):
        i = VARS_69.index(v)
        A = ax.flat[k]
        for axis, mk, c in [("capacity", "o-", "C0"), ("data", "s--", "C3")]:
            pts = [(xv, out[r]["f"][i], out[r]["rmse"][i] / br[i], lab)
                   for r, ax_, xv, lab in POINTS if ax_ == axis and r in out]
            if not pts:
                continue
            pts.sort()
            xs = np.arange(len(pts))
            A.plot(xs, [p[1] for p in pts], mk, color=c, label=f"$f_{{irr}}$ ({axis})")
            A.plot(xs, [p[2] for p in pts], mk, color=c, alpha=0.35,
                   label=f"mean-model RMSE / baseline ({axis})")
            A.set_xticks(xs, [p[3] for p in pts], fontsize=6, rotation=20)
        A.set_title(v, fontsize=9); A.set_ylim(0, 1.05)
        if k == 0:
            A.legend(frameon=False, fontsize=6)
    fig.suptitle("Does the irreducible fraction fall when the mean model gets better?  "
                 "(flat solid line = a real floor)", y=1.04)
    fig.savefig("figures/fig17_floor.png", bbox_inches="tight")
    print("\nwrote figures/fig17_floor.png")


if __name__ == "__main__":
    main()
