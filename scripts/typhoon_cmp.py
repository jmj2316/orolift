"""Same distinct warm-core 2020 cyclones as typhoon_gallery.py (selection uses ERA5 truth only),
reconstructed by every method; central pressure and peak 10 m wind, paired bootstrap over storms."""
import os, sys, json, importlib.util
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, LAT_LR, LON_LR
from orolift.varlist import VARS_69
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
BASINS = {"W. North Pacific": (105, 175, 5, 30), "North Atlantic": (260, 350, 5, 30)}
SEASON = range(6, 12)
MODELS = {"orolift50k": "runs/orolift_base/best.pt", "orolift": "runs/cmp_orolift/best.pt",
          "edsr": "runs/cmp_edsr/best.pt", "liif": "runs/cmp_liif/best.pt"}
RAD = 5.0     # degrees: window around the true center for the local minimum / peak wind
NAMES = ["input", "bilinear+proj"] + list(MODELS)


def main():
    mean, std = load_norm()
    mt = torch.tensor(mean, device=dev)[:, None, None]; st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs = np.load(f"{DATA}/resid_std.npy")
    models = {}
    for k, p in MODELS.items():
        m, _ = ev.load_model(p, rs, dev); models[k] = m
        print("loaded", k, p, "proj =", m.use_projection, flush=True)
    ip = VARS_69.index("mean_sea_level_pressure"); iu = VARS_69.index("10m_u_component_of_wind")
    iv = VARS_69.index("10m_v_component_of_wind")
    it3 = VARS_69.index("temperature_300"); it5 = VARS_69.index("temperature_500")
    ds = HR15("test"); dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4)
    den = lambda a, i: (a[:, i] * st[i] + mt[i])
    recs = []; i0 = 0
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        out = {"bilinear+proj": C.project(C.upsample(x), x)}
        for k, m in models.items():
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                out[k] = m(x, sl, sh).float()
        P = {k: (den(v, ip) / 100).cpu().numpy() for k, v in out.items()}
        W = {k: torch.hypot(den(v, iu), den(v, iv)).cpu().numpy() for k, v in out.items()}
        Py = (den(y, ip) / 100).cpu().numpy(); Wy = torch.hypot(den(y, iu), den(y, iv)).cpu().numpy()
        Px = (den(x, ip) / 100).cpu().numpy()
        Tm = ((den(y, it3) + den(y, it5)) / 2).cpu().numpy()
        for bn, (l0, l1, a0, a1) in BASINS.items():
            m_ = ((LAT_HR >= a0) & (LAT_HR <= a1))[:, None] & ((LON_HR >= l0) & (LON_HR <= l1))[None, :]
            mL = ((LAT_LR >= a0) & (LAT_LR <= a1))[:, None] & ((LON_LR >= l0) & (LON_LR <= l1))[None, :]
            for k in range(y.shape[0]):
                if int(str(ds.times[i0 + k])[5:7]) not in SEASON:
                    continue
                jj, ii = np.unravel_index(np.argmin(np.where(m_, Py[k], 1e9)), Py[k].shape)
                d2 = ((LAT_HR[:, None] - LAT_HR[jj]) ** 2 +
                      ((LON_HR[None, :] - LON_HR[ii]) * np.cos(np.deg2rad(LAT_HR[jj]))) ** 2)
                if Tm[k][d2 < 3.0 ** 2].mean() - Tm[k][(d2 > 5.0 ** 2) & (d2 < 15.0 ** 2)].mean() < 0.5:
                    continue
                loc = d2 < RAD ** 2
                r = dict(basin=bn, i=i0 + k, t=str(ds.times[i0 + k]), lat=float(LAT_HR[jj]), lon=float(LON_HR[ii]),
                         true=float(Py[k][jj, ii]), wind_true=float(Wy[k][loc].max()))
                r["input"] = dict(basin=float(np.where(mL, Px[k], 1e9).min()))
                for n in NAMES[1:]:
                    r[n] = dict(basin=float(np.where(m_, P[n][k], 1e9).min()),
                                local=float(np.where(loc, P[n][k], 1e9).min()),
                                wind=float(W[n][k][loc].max()))
                recs.append(r)
        i0 += y.shape[0]

    recs.sort(key=lambda r: r["true"])
    same = lambda a, c: (a["basin"] == c["basin"] and abs(a["i"] - c["i"]) < 28 and
                         np.hypot(a["lat"] - c["lat"], (a["lon"] - c["lon"]) * np.cos(np.deg2rad(a["lat"]))) < 30)
    ev_all = []
    for r in recs:
        if all(not same(q, r) for q in ev_all):
            ev_all.append(r)
    ev_all = [r for r in ev_all if r["true"] < 1000]
    n = len(ev_all); print("distinct storms", n)
    json.dump(ev_all, open("runs/eval/typhoon_cmp_storms.json", "w"))

    t = np.array([r["true"] for r in ev_all]); wt = np.array([r["wind_true"] for r in ev_all])
    err = {"input": np.array([r["input"]["basin"] for r in ev_all]) - t}
    errL, wr = {}, {}
    for nm in NAMES[1:]:
        err[nm] = np.array([r[nm]["basin"] for r in ev_all]) - t
        errL[nm] = np.array([r[nm]["local"] for r in ev_all]) - t
        wr[nm] = np.array([r[nm]["wind"] for r in ev_all])
    rng = np.random.default_rng(0); B = rng.integers(0, n, (20000, n))
    print(f"\n{'method':14s}{'MAE basin':>10s}{'MAE local':>10s}{'bias':>8s}{'slope':>8s}{'wind/true':>10s}{'MAEwind':>9s}")
    res = {}
    for nm in NAMES:
        e = err[nm]; sl_ = np.polyfit(t, e, 1)[0]
        row = dict(mae=float(np.abs(e).mean()), bias=float(e.mean()), slope=float(sl_))
        if nm != "input":
            row.update(mae_local=float(np.abs(errL[nm]).mean()), wind_ratio=float((wr[nm] / wt).mean()),
                       mae_wind=float(np.abs(wr[nm] - wt).mean()))
            bs = [np.polyfit(t[b], e[b], 1)[0] for b in B[:2000]]
            row["slope_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res[nm] = row
        print(f"{nm:14s}{row['mae']:10.2f}{row.get('mae_local', float('nan')):10.2f}{row['bias']:8.2f}{sl_:8.3f}"
              f"{row.get('wind_ratio', float('nan')):10.3f}{row.get('mae_wind', float('nan')):9.2f}")
    print("\npaired bootstrap of MAE difference (basin minimum), 95% CI over storms")
    pairs = [("orolift", "edsr"), ("orolift", "liif"), ("edsr", "liif"), ("orolift", "bilinear+proj"),
             ("edsr", "bilinear+proj"), ("orolift50k", "orolift")]
    res["paired"] = {}
    for a_, b_ in pairs:
        d = np.abs(err[a_]) - np.abs(err[b_]); bm = d[B].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        res["paired"][f"{a_}-{b_}"] = [float(d.mean()), float(lo), float(hi)]
        print(f"  {a_:>13s} - {b_:<14s} {d.mean():+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    wins = {nm: int((np.abs(err[nm]) <= np.min([np.abs(err[m]) for m in ["orolift", "edsr", "liif"]], 0) + 1e-9).sum())
            for nm in ["orolift", "edsr", "liif"]}
    res["wins"] = wins; res["n"] = n; print("storm-wise best (orolift/edsr/liif):", wins)
    json.dump(res, open("runs/eval/typhoon_cmp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
