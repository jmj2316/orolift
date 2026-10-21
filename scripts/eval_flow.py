"""Probabilistic evaluation of the ker(A)-constrained flow head.

The numbers that carry the scientific claim are the calibration ones -- spread/skill
and the rank histogram -- plus the member spectra, which say whether the spread is
the right *kind* of variability.  CRPS is reported because it is a proper score and
guards against buying calibration with useless spread, not because the claim rests
on it.
"""
import argparse, json, os, sys, importlib.util
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, area_weights, LAT_HR, LON_HR
from orolift.metrics import zonal_spectrum, lat_weighted_rmse
from orolift.models.flow import FlowHead, FlowSampler
from orolift.varlist import VARS_69, LEVELS, ATMOS_PREFIXES
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

KEY = ["2m_temperature", "mean_sea_level_pressure", "geopotential_500",
       "temperature_850", "u_component_of_wind_850", "specific_humidity_700"]
dev = "cuda"


def crps_ensemble(ens, obs, w):
    M = ens.shape[1]
    t1 = (ens - obs[:, None]).abs().mean(1)
    t2 = (ens[:, :, None] - ens[:, None, :]).abs().sum(dim=(1, 2)) / (2 * M * (M - 1))
    return (w * (t1 - t2)).mean(dim=(-2, -1)).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/flow_base/best.pt")
    ap.add_argument("--det", default="runs/orolift_base/best.pt")
    ap.add_argument("--hetero", default="runs/hetero_base/best.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--ode_steps", type=int, default=16)
    ap.add_argument("--tag", default="flow_base")
    a = ap.parse_args()

    mean, std = load_norm()
    mt = torch.tensor(mean, device=dev)[:, None, None]
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs_np = np.load(f"{DATA}/resid_std.npy")
    rs = torch.tensor(rs_np, device=dev)[None, :, None, None]
    w = torch.as_tensor(area_weights(), device=dev, dtype=torch.float32)[:, None]

    det, _ = ev.load_model(a.det, rs_np, dev); det.eval()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = FlowHead(dim=d["cfg"]["dim"], depth=d["cfg"]["depth"], resid_std=rs_np,
                   sigma_cond=d["cfg"].get("sigma_cond", False)).to(dev)
    pd = dict(net.named_parameters())
    with torch.no_grad():
        for k, v in d["ema"].items():
            pd[k].copy_(v.to(dev))
    net.eval()
    print(f"flow {a.ckpt} step {d['step']}   members {a.members}   ode steps {a.ode_steps}")

    het = None
    if os.path.exists(a.hetero):
        from orolift.models.hetero import VarHead
        dh = torch.load(a.hetero, map_location="cpu", weights_only=False)
        het = VarHead(dim=dh["cfg"]["dim"], depth=dh["cfg"]["depth"]).to(dev)
        ph = dict(het.named_parameters())
        with torch.no_grad():
            for k, v in dh["ema"].items():
                ph[k].copy_(v.to(dev))
        het.eval()
        print(f"hetero {a.hetero} step {dh['step']}")
    smp = FlowSampler(net, C, det, het)
    assert not net.sigma_cond or het is not None, "sigma-conditioned flow needs the variance head"

    dl = torch.utils.data.DataLoader(HR15("test", limit=a.n), batch_size=a.bs, num_workers=4)
    M = a.members
    acc = dict(crps=0, rmse_mean=0, rmse_det=0, spread=0, cons=0, n=0,
               sp_truth=0, sp_det=0, sp_mem=0, sp_mean=0, sig_het=0, sig_flow=0, nb=0)
    ranks = torch.zeros(len(VARS_69), M + 1, device=dev)
    sig_map = 0; sig_map_h = 0
    for y in dl:
        y = y.to(dev); x = C.coarsen(y)
        ens = smp.sample(x, sl, sh, n_members=M, steps=a.ode_steps)     # (B,M,V,H,W)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            yd = det(x, sl, sh).float()
        eP, yP, dP = ens * st + mt, y * st + mt, yd * st + mt
        mP = eP.mean(1)
        B = y.shape[0]
        acc["crps"] = acc["crps"] + crps_ensemble(eP, yP, w) * B
        acc["rmse_mean"] = acc["rmse_mean"] + lat_weighted_rmse(mP, yP) * B
        acc["rmse_det"] = acc["rmse_det"] + lat_weighted_rmse(dP, yP) * B
        sd = eP.std(1)
        acc["spread"] = acc["spread"] + torch.sqrt((w * sd ** 2).mean(dim=(-2, -1))).mean(0) * B
        acc["cons"] += float(C.consistency_error(ens.reshape(-1, *ens.shape[2:]),
                                                 x.repeat_interleave(M, 0)).mean()) * B
        acc["sp_truth"] = acc["sp_truth"] + zonal_spectrum(yP) * B
        acc["sp_det"] = acc["sp_det"] + zonal_spectrum(dP) * B
        acc["sp_mem"] = acc["sp_mem"] + zonal_spectrum(eP[:, 0]) * B
        acc["sp_mean"] = acc["sp_mean"] + zonal_spectrum(mP) * B
        r = (eP < yP[:, None]).sum(1)                                   # 0..M
        for v in range(len(VARS_69)):
            ranks[v] += torch.bincount(r[:, v].reshape(-1), minlength=M + 1).float()
        sig_map = sig_map + sd.sum(0)
        acc["sig_flow"] = acc["sig_flow"] + torch.sqrt((w * (sd / (st * rs[0])) ** 2
                                                        ).mean(dim=(-2, -1))).sum(0)
        if het is not None:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                sh_ = torch.exp(het(yd / rs, sh).float())
            sig_map_h = sig_map_h + (sh_ * rs * st).sum(0)
            acc["sig_het"] = acc["sig_het"] + (w * sh_).mean(dim=(-2, -1)).sum(0)
        acc["n"] += B
    n = acc["n"]
    R = {k: (v / n if torch.is_tensor(v) else v / n) for k, v in acc.items() if k != "n"}
    sig_map = (sig_map / n).cpu().numpy(); 
    if het is not None:
        sig_map_h = (sig_map_h / n).cpu().numpy()

    corr = np.sqrt((M + 1) / M)
    print(f"\n{'variable':26s}{'CRPS':>11s}{'RMSE ens':>10s}{'RMSE det':>10s}"
          f"{'spread':>10s}{'sp/skill':>10s}")
    rows = {}
    for v in KEY:
        i = VARS_69.index(v)
        ss = float(R["spread"][i] * corr / R["rmse_mean"][i])
        print(f"{v:26s}{R['crps'][i]:11.4g}{R['rmse_mean'][i]:10.4g}{R['rmse_det'][i]:10.4g}"
              f"{R['spread'][i]:10.4g}{ss:10.3f}")
        rows[v] = dict(crps=float(R["crps"][i]), rmse_ens=float(R["rmse_mean"][i]),
                       rmse_det=float(R["rmse_det"][i]), spread=float(R["spread"][i]), ss=ss)
    ss_all = (R["spread"] * corr / R["rmse_mean"]).cpu().numpy()
    print(f"\nall 69 vars: mean spread/skill {ss_all.mean():.3f}  "
          f"(median {np.median(ss_all):.3f}, 10-90% {np.percentile(ss_all,10):.2f}-{np.percentile(ss_all,90):.2f})")
    print(f"member consistency ||A y - x|| = {R['cons']:.3e}")
    print(f"ensemble-mean RMSE / deterministic RMSE = "
          f"{float((R['rmse_mean']/R['rmse_det']).mean()):.4f}")
    if het is not None:
        a_ = R["sig_flow"].cpu().numpy(); b_ = R["sig_het"].cpu().numpy()
        print(f"\nsigma (units of the free-baseline residual): flow {a_.mean():.3f}  "
              f"hetero {b_.mean():.3f}   ratio {np.mean(a_/np.maximum(b_,1e-9)):.3f}  "
              f"corr across 69 vars {np.corrcoef(a_, b_)[0,1]:.3f}")

    kk = np.arange(R["sp_truth"].shape[1])
    out = dict(rows=rows, ss_all=ss_all.tolist(), cons=R["cons"],
               ranks=ranks.cpu().numpy().tolist(), vars=VARS_69,
               sigma_flow=R["sig_flow"].cpu().numpy().tolist(),
               sigma_het=(R["sig_het"].cpu().numpy().tolist() if het is not None else None))
    os.makedirs("runs/eval", exist_ok=True)
    json.dump(out, open(f"runs/eval/{a.tag}_prob.json", "w"))

    # ---------------------------------------------------------------- figure
    iu, iq = VARS_69.index("u_component_of_wind_850"), VARS_69.index("specific_humidity_700")
    np.savez(f"runs/eval/{a.tag}_figdata.npz", ss_all=ss_all, ranks=ranks.cpu().numpy(), kk=kk,
             sp=np.stack([R[k][iu].cpu().numpy() for k in ["sp_truth", "sp_det", "sp_mean", "sp_mem"]]),
             sig_q=sig_map[iq], lsm=sh[0, 3].cpu().numpy(), cons=R["cons"])
    fig = plt.figure(figsize=(10.2, 7.4))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    NAME = {"geopotential": "Geopotential", "temperature": "Temperature", "u_component_of_wind": "Zonal wind",
            "v_component_of_wind": "Meridional wind", "specific_humidity": "Humidity"}
    cols = ["#1D2B55", "#3F6BBE", "#7F9FD3", "#D64550", "#E8A33D"]
    for (p, c) in zip(ATMOS_PREFIXES, cols):
        vals = [ss_all[VARS_69.index(f"{p}_{l}")] for l in LEVELS]
        ax.plot(vals, LEVELS, "o-", ms=3, lw=1.2, color=c, label=NAME.get(p, p))
    ax.axvline(1.0, color="0.2", lw=0.9, ls="--")
    ax.invert_yaxis(); ax.set_yscale("log"); ax.set_yticks(LEVELS, [str(l) for l in LEVELS], fontsize=7.5)
    ax.minorticks_off(); ax.set_xlabel("spread / skill"); ax.set_ylabel("pressure (hPa)")
    ax.legend(fontsize=7.5, loc="lower left", handlelength=1.4, borderaxespad=0.2)
    ax.set_title("Calibration", fontsize=9.5); S.tag(ax, "a", dx=0.0, dy=1.1)
    RT = {"2m_temperature": "2 m temperature", "mean_sea_level_pressure": "Sea-level pressure",
          "geopotential_500": "500 hPa geopotential"}
    for j, v in enumerate(KEY[:3]):
        axr = fig.add_subplot(gs[0, 1 + j])
        h = ranks[VARS_69.index(v)].cpu().numpy(); h = h / h.sum()
        axr.bar(np.arange(M + 1), h, color=S.PAL["liif"], edgecolor="0.2", lw=0.4, width=0.85)
        axr.axhline(1 / (M + 1), color=S.ACCENT, lw=1.2, ls="--")
        axr.set_title(RT[v], fontsize=9.5); axr.set_xlabel("rank of truth in ensemble")
        axr.set_yticks([]); axr.spines[["top", "right", "left"]].set_visible(False)
        if j == 0:
            axr.set_ylabel("frequency"); S.tag(axr, "b", dx=0.0, dy=1.1)
    axs = fig.add_subplot(gs[1, 0:2])
    for k, lab, c, ls, lw in [("sp_truth", "ERA5 1.5°", S.PAL["truth"], "-", 1.9),
                              ("sp_det", "Deterministic OroLIFT", S.PAL["liif"], "-", 1.2),
                              ("sp_mean", "Ensemble mean", S.PAL["bilinear+proj"], "--", 1.2),
                              ("sp_mem", "Single member", S.ACCENT, "-", 1.5)]:
        axs.loglog(kk[1:], R[k][iu][1:].cpu().numpy(), color=c, ls=ls, lw=lw, label=lab)
    axs.axvline(32, color="0.4", lw=0.8, ls=":"); axs.axvspan(32, kk[-1] + 2, color="0.5", alpha=0.07, lw=0)
    axs.set_xlim(1, kk[-1] + 2)
    axs.set_xlabel("zonal wavenumber $k$"); axs.set_ylabel("zonal power")
    axs.legend(fontsize=8, loc="lower left"); axs.set_title("850 hPa zonal wind: members restore the spectral tail", fontsize=9.5)
    S.tag(axs, "c", dx=0.0, dy=1.1)
    axm = fig.add_subplot(gs[1, 2:4])
    sh_ = 120
    Z = np.roll(sig_map[iq] * 1e3, sh_, axis=1); Lm = np.roll(sh[0, 3].cpu().numpy(), sh_, axis=1)
    lo1 = np.roll(LON_HR, sh_); lo1 = np.where(lo1 > 180, lo1 - 360, lo1)
    im = axm.pcolormesh(lo1, LAT_HR, Z, cmap="magma_r", shading="auto", rasterized=True)
    axm.contour(lo1, LAT_HR, Lm, levels=[0.5], colors="w", linewidths=0.4)
    axm.set_xlim(-180, 180); axm.set_ylim(-90, 90); axm.set_aspect("equal")
    axm.set_xticks([-120, 0, 120], ["120°W", "0°", "120°E"]); axm.set_yticks([-60, 0, 60], ["60°S", "0°", "60°N"])
    cb = fig.colorbar(im, ax=axm, shrink=0.8, pad=0.02); cb.set_label(r"g kg$^{-1}$")
    axm.set_title("Conditional spread of 700 hPa humidity", fontsize=9.5); S.tag(axm, "d", dx=0.0, dy=1.1)
    fig.savefig("figures/fig12_flow_new.png"); plt.close(fig)
    print("\nwrote figures/fig12_flow_new.png")


if __name__ == "__main__":
    main()
