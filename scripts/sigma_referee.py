"""Which variance estimator is right, and is the disagreement just sampling noise?

Two tests:

(a) Referee.  The realized deterministic error, RMS_t[(y - y_det)] at each gridpoint,
    is a model-free estimate of the same conditional standard deviation both heads
    are predicting.  Whichever head correlates with it spatially is the better one.
    (It conflates conditional variance with any conditional bias of y_det, but it
    referees both estimators on identical terms.)

(b) Sampling noise.  Re-estimate the flow variance with many more members.  The
    relative error of an M-member variance is sqrt(2/(M-1)) per sample, averaged
    over n times -- so if 16 members were the problem, 48 must move the answer.
"""
import argparse, os, sys, json, importlib.util
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, area_weights
from orolift.models.flow import FlowHead, FlowSampler
from orolift.models.hetero import VarHead
from orolift.varlist import VARS_69, LEVELS
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
KEY = ["2m_temperature", "mean_sea_level_pressure", "geopotential_500", "temperature_850",
       "u_component_of_wind_850", "specific_humidity_700", "specific_humidity_925"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="runs/flow_base/best.pt")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--members", type=int, default=48)
    ap.add_argument("--ode_steps", type=int, default=12)
    ap.add_argument("--tag", default=None, help="write only runs/eval/sigma_referee_<tag>.json")
    a = ap.parse_args()

    mean, std = load_norm()
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs_np = np.load(f"{DATA}/resid_std.npy")
    rs = torch.tensor(rs_np, device=dev)[None, :, None, None]
    det, _ = ev.load_model("runs/orolift_base/best.pt", rs_np, dev); det.eval()

    d = torch.load(a.flow, map_location="cpu", weights_only=False)
    print(f"flow ckpt {a.flow}")
    net = FlowHead(dim=d["cfg"]["dim"], depth=d["cfg"]["depth"], resid_std=rs_np,
                   sigma_cond=d["cfg"].get("sigma_cond", False)).to(dev)
    pd = dict(net.named_parameters())
    with torch.no_grad():
        for k, v in d["ema"].items():
            pd[k].copy_(v.to(dev))
    net.eval()

    dh = torch.load("runs/hetero_base/best.pt", map_location="cpu", weights_only=False)
    het = VarHead(dim=dh["cfg"]["dim"], depth=dh["cfg"]["depth"]).to(dev)
    ph = dict(het.named_parameters())
    with torch.no_grad():
        for k, v in dh["ema"].items():
            ph[k].copy_(v.to(dev))
    het.eval()
    smp = FlowSampler(net, C, det, het)

    V, H, W_ = len(VARS_69), len(LAT_HR), len(LON_HR)
    zeros = lambda: torch.zeros(V, H, W_, device=dev, dtype=torch.float64)
    v_flow16, v_flowM, v_het, v_real, den = zeros(), zeros(), zeros(), zeros(), zeros()
    n = 0
    dl = torch.utils.data.DataLoader(HR15("test", limit=a.n), batch_size=a.bs, num_workers=4)
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        ens = smp.sample(x, sl, sh, n_members=a.members, steps=a.ode_steps)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            yd = det(x, sl, sh).float()
            s = torch.exp(het(yd / rs, sh).float())
        base = C.project(C.upsample(x), x)
        v_flowM += (ens.var(1) * st ** 2).sum(0).double()
        v_flow16 += (ens[:, :16].var(1) * st ** 2).sum(0).double()
        v_het += ((s * rs * st) ** 2).sum(0).double()
        v_real += (((y - yd) * st) ** 2).sum(0).double()      # realized, model-free
        den += (((y - base) * st) ** 2).sum(0).double()
        n += y.shape[0]
    A = {k: (v / n).cpu().numpy() for k, v in
         dict(flowM=v_flowM, flow16=v_flow16, het=v_het, real=v_real, den=den).items()}
    if a.tag is None:
        np.savez("runs/sigma_referee.npz", **A, vars=np.array(VARS_69))

    Wt = np.broadcast_to(area_weights()[:, None], (H, W_))
    def sc(a_, b_, i, msk):                 # weighted spatial correlation over valid points
        x1, x2, w_ = a_[i][msk], b_[i][msk], Wt[msk]
        m1 = (x1 * w_).sum() / w_.sum(); m2 = (x2 * w_).sum() / w_.sum()
        c = (w_ * (x1 - m1) * (x2 - m2)).sum()
        return float(c / np.sqrt((w_ * (x1 - m1) ** 2).sum() * (w_ * (x2 - m2) ** 2).sum()))

    print(f"n={n} samples, {a.members} members\n")
    print(f"{'variable':26s}{'flow16~flow48':>15s}{'flow~realized':>15s}"
          f"{'het~realized':>14s}{'flow~het':>10s}")
    rows = {}
    for v in KEY:
        i = VARS_69.index(v)
        thr = 0.15 * float((A["den"][i] * Wt).sum() / Wt.sum())
        m = A["den"][i] >= thr
        r = (sc(A["flow16"], A["flowM"], i, m), sc(A["flowM"], A["real"], i, m),
             sc(A["het"], A["real"], i, m), sc(A["flowM"], A["het"], i, m))
        rows[v] = r
        print(f"{v:26s}{r[0]:15.3f}{r[1]:15.3f}{r[2]:14.3f}{r[3]:10.3f}")
    allr = []
    for i in range(V):
        thr = 0.15 * float((A["den"][i] * Wt).sum() / Wt.sum())
        m = A["den"][i] >= thr
        if m.sum() < 500:
            continue
        allr.append((sc(A["flow16"], A["flowM"], i, m), sc(A["flowM"], A["real"], i, m),
                     sc(A["het"], A["real"], i, m), sc(A["flowM"], A["het"], i, m)))
    allr = np.array(allr)
    lab = ["flow16 ~ flow48 (sampling noise)", "flow ~ realized (referee)",
           "hetero ~ realized (referee)", "flow ~ hetero"]
    print(f"\nmedian over {len(allr)} variables:")
    for k, l in enumerate(lab):
        print(f"   {l:36s} {np.median(allr[:,k]):6.3f}   "
              f"(10-90%  {np.percentile(allr[:,k],10):.2f} - {np.percentile(allr[:,k],90):.2f})")
    json.dump({"rows": rows, "median": np.median(allr, 0).tolist()},
              open("runs/eval/sigma_referee.json" if a.tag is None else f"runs/eval/sigma_referee_{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
