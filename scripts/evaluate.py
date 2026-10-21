"""Full evaluation of a trained checkpoint (or the free baseline) on the test split.

`--input synthetic` feeds A.y_HR (the training-time operator).
`--input real` feeds the *independently produced* 5.625-degree files, i.e. the
operator-mismatch / blind-degradation setting.
"""
import argparse, datetime as dt, json, os, sys
import numpy as np, torch, h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener
from orolift.metrics import (lat_weighted_rmse, lat_weighted_acc, spectral_ratio,
                             hydrostatic_residual, geostrophic_residual,
                             quantile_error, terrain_stratified_rmse)
from orolift.models.baselines import build_model
from orolift.varlist import VARS_69

KEY = ["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure",
       "geopotential_500", "temperature_850", "u_component_of_wind_850",
       "specific_humidity_700", "temperature_1000"]
LR5 = os.path.join(os.environ.get("OROLIFT_SRC_5P625", ""), "test")   # independently archived 5.625-degree ERA5, 2020 = test year


class RealLR(torch.utils.data.Dataset):
    """Stored 5.625-degree fields for the test year, normalized the same way."""

    def __init__(self, times, mean, std):
        self.files = sorted(f"{LR5}/{f}" for f in os.listdir(LR5))[: len(times)]
        self.mean, self.std = mean, std
        for i in (0, len(self.files) - 1):
            with h5py.File(self.files[i], "r") as f:
                t = f["input"]["time"][()].decode()
            assert t == str(times[i]), f"{t} vs {times[i]}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with h5py.File(self.files[i], "r") as f:
            g = f["input"]
            a = np.stack([g[v][()].T for v in VARS_69]).astype(np.float32)
        return torch.from_numpy((a - self.mean[:, None, None]) / self.std[:, None, None])


def load_model(ckpt, resid_std, dev):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = d["cfg"]
    m = build_model(c.get("arch", "orolift"), dim=c["dim"], depth=c["depth"],
                    render_width=c["render_width"], render_depth=c["render_depth"],
                    direct_dim=c["direct_dim"], resid_std=resid_std,
                    use_projection=not c["no_projection"],
                    use_orography=not c["no_orography"])
    # Load EMA *parameters* only.  Older checkpoints stored an EMA of the whole
    # state_dict, which had slowly corrupted the coarsener's constant buffers; the
    # module's own freshly built buffers are the exact ones.
    sd, pnames = dict(m.state_dict()), set(dict(m.named_parameters()))
    m.load_state_dict({k: (v.to(sd[k].dtype) if k in pnames else sd[k])
                       for k, v in d["ema"].items() if k in sd}, strict=False)
    return m.to(dev).eval(), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="omit to evaluate the free baseline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--input", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--proj_dtype", default="float32", choices=["float32", "float64"])
    a = ap.parse_args()
    dev = "cuda"

    mean, std = load_norm()
    mean_t = torch.tensor(mean, device=dev)[:, None, None]
    std_t = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    C64 = SeparableCoarsener("conservative", dtype=torch.float64).to(dev)   # metric only
    sh, sl = static_features(dev)
    sh, sl = sh[None], sl[None]
    rs = np.load(f"{DATA}/resid_std.npy")

    hr = HR15(a.split, limit=a.n)
    dl = torch.utils.data.DataLoader(hr, batch_size=a.bs, num_workers=6)
    dlx = None
    if a.input == "real":
        rl = RealLR(hr.times, mean, std)
        dlx = torch.utils.data.DataLoader(rl, batch_size=a.bs, num_workers=6)

    clim = None
    cp = f"{DATA}/clim_doy.npy"
    if os.path.exists(cp):
        cd = np.load(cp, mmap_mode="r")
        doy = np.array([dt.date(*map(int, str(s)[:10].split("-"))).timetuple().tm_yday - 1
                        for s in hr.times])

    model = None
    if a.ckpt:
        model, cfg = load_model(a.ckpt, rs, dev)
        model.proj_dtype = getattr(torch, a.proj_dtype)
        print(f"loaded {a.ckpt}  arch={cfg.get('arch','orolift')}  "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    S = {k: [] for k in ["rmse", "acc", "spec", "cons", "hyd", "geo"]}
    T = {"hyd": [], "geo": []}
    terr, P, Q = [], [], []
    i0 = 0
    it = zip(dl, dlx) if dlx is not None else ((y, None) for y in dl)
    for y, xreal in it:
        y = y.to(dev, non_blocking=True)
        x = xreal.to(dev) if xreal is not None else C.coarsen(y)
        if model is None:
            p = C.project(C.upsample(x), x)
            S["cons"].append(C.consistency_error(p, x).mean(0).cpu())
        else:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                praw = model(x, sl, sh)
            # measure consistency in the dtype the model produced: casting to fp32
            # first would report fp32 storage roundoff, not the projection's error
            Cm = C64 if praw.dtype == torch.float64 else C
            S["cons"].append(Cm.consistency_error(praw, x.to(praw.dtype)).mean(0).float().cpu())
            p = praw.float()
        pp, yy = p * std_t + mean_t, y * std_t + mean_t
        S["rmse"].append(lat_weighted_rmse(pp, yy).cpu())
        S["spec"].append(spectral_ratio(pp, yy).cpu())
        S["hyd"].append(hydrostatic_residual(pp).mean(0).cpu())
        S["geo"].append(geostrophic_residual(pp).mean(0).cpu())
        T["hyd"].append(hydrostatic_residual(yy).mean(0).cpu())
        T["geo"].append(geostrophic_residual(yy).mean(0).cpu())
        terr.append(terrain_stratified_rmse(pp, yy, sh[0, 1] * 1000.0).cpu())
        if clim is not None or os.path.exists(cp):
            cb = torch.as_tensor(np.asarray(cd[doy[i0:i0 + y.shape[0]]], np.float32), device=dev)
            S["acc"].append(lat_weighted_acc(pp, yy, cb * std_t + mean_t).cpu())
        P.append(pp[:, [VARS_69.index(v) for v in KEY]].cpu())
        Q.append(yy[:, [VARS_69.index(v) for v in KEY]].cpu())
        i0 += y.shape[0]

    R = {k: torch.stack(v).mean(0) for k, v in S.items() if v}
    Tr = {k: torch.stack(v).mean(0) for k, v in T.items()}
    terr = torch.stack(terr).mean(0)
    q99 = quantile_error(torch.cat(P), torch.cat(Q), 0.99)

    tag = a.tag or (os.path.basename(os.path.dirname(a.ckpt)) if a.ckpt else "baseline")
    print(f"\n=== {tag} | split={a.split} n={len(hr)} | input={a.input} ===")
    hdr = f"{'variable':26s}{'RMSE':>12s}{'ACC':>9s}{'spec(k>=32)':>13s}{'P99 err':>12s}"
    print(hdr)
    for v in KEY:
        i = VARS_69.index(v)
        acc = f"{R['acc'][i]:9.4f}" if "acc" in R else f"{'-':>9s}"
        print(f"{v:26s}{R['rmse'][i]:12.4g}{acc}{R['spec'][i]:13.3f}{q99[KEY.index(v)]:12.4g}")
    print(f"\nconsistency ||A y - x||  = {R['cons'].mean():.3e}")
    print(f"hydrostatic  model={R['hyd'].mean():.4g}   truth={Tr['hyd'].mean():.4g}")
    print(f"ageostrophic model={R['geo'].mean():.4g}   truth={Tr['geo'].mean():.4g}")
    print("\nRMSE by |sub-grid orography|: [0-100) [100-400) [400+) m")
    for v in ["2m_temperature", "temperature_1000", "u_component_of_wind_850"]:
        i = VARS_69.index(v)
        print(f"   {v:24s}" + "".join(f"{terr[b][i]:11.4g}" for b in range(3)))

    os.makedirs("runs/eval", exist_ok=True)
    out = f"runs/eval/{tag}_{a.split}_{a.input}.json"
    json.dump({"rmse": R["rmse"].tolist(), "spec": R["spec"].tolist(),
               "acc": R["acc"].tolist() if "acc" in R else None,
               "cons": float(R["cons"].mean()), "hyd": R["hyd"].tolist(),
               "geo": R["geo"].tolist(), "hyd_truth": Tr["hyd"].tolist(),
               "geo_truth": Tr["geo"].tolist(), "terrain": terr.tolist(),
               "q99_key": {v: float(q99[i]) for i, v in enumerate(KEY)},
               "vars": VARS_69}, open(out, "w"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
