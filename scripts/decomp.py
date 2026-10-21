"""Empirical range/null-space decomposition of the downscaling error.

For any output yhat and truth y with coarse field A y, let Pi be the exact projection onto
{z : A z = A y}.  Then, in the area-weighted norm,
    ||yhat - y||^2 = ||yhat - Pi yhat||^2 (range, set by the coarse field) + ||Pi yhat - y||^2 (null space).
Part A (lead 0, x = A y): unconstrained outputs, to measure the range share the projection removes.
Part B: x_f = A y(t - lag), a lagged-analysis (persistence) stress test of the claim that a consistent
downscaler cannot change the resolved-scale term.  This is NOT a forecast.
"""
import argparse, json, os, sys, importlib.util
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, static_features, DATA
from orolift.grids import SeparableCoarsener, area_weights
from orolift.varlist import VARS_69
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
MODELS = {"orolift": "runs/cmp_orolift/best.pt", "edsr": "runs/cmp_edsr/best.pt",
          "liif": "runs/cmp_liif/best.pt", "noproj": "runs/abl_noproj/best.pt"}
LAGS = [0, 1, 4, 12]          # x 6 h
ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.7]   # fraction of the 1-day persistence error added to x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    C64 = SeparableCoarsener("conservative", dtype=torch.float64).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs = np.load(f"{DATA}/resid_std.npy")
    w = torch.tensor(area_weights(), device=dev, dtype=torch.float64)[:, None]
    V = len(VARS_69)
    models = {}
    for k, p in MODELS.items():
        m, _ = ev.load_model(p, rs, dev); models[k] = m
        print("loaded", k, p, "use_projection =", m.use_projection, flush=True)

    def decompose(yh, y):
        yh, y = yh.double(), y.double()
        yp = C64.project(yh, C64.coarsen(y))
        f = lambda e: (w * e ** 2).mean(dim=(-2, -1)).sum(0)
        return torch.stack([f(yh - y), f(yh - yp), f(yp - y)])         # (3, V): total, range, null

    def fwd(name, x, proj):
        m = models[name]; m.use_projection = proj
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            return m(x, sl, sh).float()

    hr = HR15("test", limit=a.limit)
    idx = list(range(max(LAGS), len(hr), a.stride))
    n = len(idx); print("samples", n, flush=True)
    zero = lambda: torch.zeros(3, V, device=dev, dtype=torch.float64)
    A = {k: zero() for k in ["bilinear"] + list(MODELS)}
    B = {(lag, k): zero() for lag in LAGS for k in ["bilinear+proj"] + [m for m in MODELS if m != "noproj"]}
    Cc = {(al, k): zero() for al in ALPHAS for k in ["bilinear+proj"] + [m for m in MODELS if m != "noproj"]}
    ident = 0.0
    for b in range(0, n, a.bs):
        ch = idx[b:b + a.bs]
        y = torch.stack([hr[i] for i in ch]).to(dev)
        x = C.coarsen(y)
        raws = {"bilinear": C.upsample(x)}
        for k in MODELS:
            raws[k] = fwd(k, x, False)
        for k, r in raws.items():
            d = decompose(r, y); A[k] += d
            ident = max(ident, float(((d[0] - d[1] - d[2]).abs() / d[0]).max()))
        for lag in LAGS:
            yl = y if lag == 0 else torch.stack([hr[i - lag] for i in ch]).to(dev)
            xf = C.coarsen(yl)
            B[(lag, "bilinear+proj")] += decompose(C.project(C.upsample(xf), xf), y)
            for k in MODELS:
                if k != "noproj":
                    B[(lag, k)] += decompose(fwd(k, xf, True), y)
        yl = torch.stack([hr[i - 4] for i in ch]).to(dev)
        for al in ALPHAS:
            xf = x + al * (C.coarsen(yl) - x)
            Cc[(al, "bilinear+proj")] += decompose(C.project(C.upsample(xf), xf), y)
            for k in MODELS:
                if k != "noproj":
                    Cc[(al, k)] += decompose(fwd(k, xf, True), y)
        if (b // a.bs) % 5 == 0:
            print(f"{b + len(ch)}/{n}", flush=True)
    out = {"n": n, "vars": VARS_69, "ident_max_rel": ident,
           "A": {k: (v / n).cpu().tolist() for k, v in A.items()},
           "B": {f"{lag}|{k}": (v / n).cpu().tolist() for (lag, k), v in B.items()},
           "C": {f"{al}|{k}": (v / n).cpu().tolist() for (al, k), v in Cc.items()}}
    os.makedirs("runs/eval", exist_ok=True)
    json.dump(out, open("runs/eval/decomp.json", "w"))
    print("identity max relative residual", ident)
    print("wrote runs/eval/decomp.json")


if __name__ == "__main__":
    main()
