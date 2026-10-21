"""Is the recoverable part of sub-grid structure the *balanced* part?

Hypothesis: fine structure that is slaved to the large scale through balance
(rotational / vortical) should be reconstructible from a coarse state, while the
unbalanced part (divergent -- gravity waves, convection, ageostrophic outflow)
should not be, no matter how good the model is.

Test: Helmholtz-decompose the wind into relative vorticity and divergence on the
sphere, and compare how much of each the downscaler recovers, level by level.
"""
import os, sys, json, importlib.util
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, area_weights
from orolift.varlist import VARS_69, LEVELS
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
A_EARTH = 6371.0e3
LATLIM = 80.0                     # drop the polar rows: 1/cos(phi) blows up


def vort_div(u, v):
    """Relative vorticity and horizontal divergence on the regular lat-lon sphere."""
    lat = torch.as_tensor(np.deg2rad(LAT_HR), device=u.device, dtype=u.dtype)
    cos = torch.cos(lat).clamp_min(1e-6)[:, None]
    dphi = float(np.deg2rad(LAT_HR[1] - LAT_HR[0]))
    dlam = float(np.deg2rad(LON_HR[1] - LON_HR[0]))
    ddl = lambda f: (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) / (2 * dlam)
    ddp = lambda f: torch.gradient(f, spacing=dphi, dim=-2)[0]
    zeta = (ddl(v) - ddp(u * cos)) / (A_EARTH * cos)
    div = (ddl(u) + ddp(v * cos)) / (A_EARTH * cos)
    return zeta, div


def main():
    mean, std = load_norm()
    mt = torch.tensor(mean, device=dev)[:, None, None]
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    model, _ = ev.load_model("runs/orolift_base/best.pt", np.load(f"{DATA}/resid_std.npy"), dev)

    ui = [VARS_69.index(f"u_component_of_wind_{l}") for l in LEVELS]
    vi = [VARS_69.index(f"v_component_of_wind_{l}") for l in LEVELS]
    keep = torch.as_tensor(np.abs(LAT_HR) <= LATLIM, device=dev)
    w = torch.as_tensor(area_weights(), device=dev, dtype=torch.float32)[:, None][keep]

    dl = torch.utils.data.DataLoader(HR15("test", limit=512), batch_size=16, num_workers=4)
    acc = {k: 0.0 for k in ["zt", "zb", "zo", "dt", "db", "do", "n"]}
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            p = model(x, sl, sh).float()
        b = C.project(C.upsample(x), x)
        out = {}
        for tag, f in [("t", y), ("b", b), ("o", p)]:
            U = f[:, ui] * st[ui] + mt[ui]
            V = f[:, vi] * st[vi] + mt[vi]
            z, d = vort_div(U, V)
            out["z" + tag], out["d" + tag] = z[..., keep, :], d[..., keep, :]
        for k in ["zt", "zb", "zo", "dt", "db", "do"]:
            acc[k] = acc[k] + out[k].double() * 0 if False else acc[k]
        # accumulate error and variance sums (weighted, per level)
        for f, tgt in [("b", "zb"), ("o", "zo")]:
            e = ((out[tgt] - out["zt"]) ** 2 * w).mean(dim=(-2, -1)).sum(0)
            acc.setdefault("ez" + f, 0); acc["ez" + f] = acc["ez" + f] + e.double()
        for f, tgt in [("b", "db"), ("o", "do")]:
            e = ((out[tgt] - out["dt"]) ** 2 * w).mean(dim=(-2, -1)).sum(0)
            acc.setdefault("ed" + f, 0); acc["ed" + f] = acc["ed" + f] + e.double()
        for nm, tgt in [("vz", "zt"), ("vd", "dt")]:
            s = ((out[tgt] - (out[tgt] * w).mean(dim=(-2, -1), keepdim=True)) ** 2 * w).mean(dim=(-2, -1)).sum(0)
            acc.setdefault(nm, 0); acc[nm] = acc[nm] + s.double()
        acc["n"] += y.shape[0]

    n = acc["n"]
    rz_b, rz_o = torch.sqrt(acc["ezb"] / n), torch.sqrt(acc["ezo"] / n)
    rd_b, rd_o = torch.sqrt(acc["edb"] / n), torch.sqrt(acc["edo"] / n)
    sz, sd = torch.sqrt(acc["vz"] / n), torch.sqrt(acc["vd"] / n)

    print(f"{'level':>7s} | {'VORTICITY (balanced)':^34s} | {'DIVERGENCE (unbalanced)':^34s}")
    print(f"{'hPa':>7s} | {'base':>9s}{'OroLIFT':>9s}{'skill':>8s}{'frac':>8s} | "
          f"{'base':>9s}{'OroLIFT':>9s}{'skill':>8s}{'frac':>8s}")
    rows = []
    for k, l in enumerate(LEVELS):
        # skill = 1 - RMSE_model/RMSE_baseline ; frac = variance explained beyond baseline
        skz = 1 - float(rz_o[k] / rz_b[k]); skd = 1 - float(rd_o[k] / rd_b[k])
        fz = 1 - float(rz_o[k] / sz[k]) ** 2 if float(rz_o[k]) < float(sz[k]) else 0.0
        fd = 1 - float(rd_o[k] / sd[k]) ** 2 if float(rd_o[k]) < float(sd[k]) else 0.0
        print(f"{l:7d} | {float(rz_b[k]):9.3e}{float(rz_o[k]):9.3e}{skz:8.3f}{fz:8.3f} | "
              f"{float(rd_b[k]):9.3e}{float(rd_o[k]):9.3e}{skd:8.3f}{fd:8.3f}")
        rows.append(dict(level=l, zeta_skill=skz, div_skill=skd, zeta_frac=fz, div_frac=fd))
    zs = np.mean([r["zeta_skill"] for r in rows]); dsk = np.mean([r["div_skill"] for r in rows])
    zf = np.mean([r["zeta_frac"] for r in rows]); df = np.mean([r["div_frac"] for r in rows])
    print(f"\nmean over 13 levels:  RMSE reduction vs baseline  vorticity {zs:.3f}   divergence {dsk:.3f}")
    print(f"                      variance of the field explained  vorticity {zf:.3f}   divergence {df:.3f}")
    json.dump(rows, open("runs/balance_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
