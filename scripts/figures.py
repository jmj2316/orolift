"""Figures for OroLIFT.

fig1  case study over the Himalaya / Tibetan Plateau (2m temperature)
fig2  zonal power spectra -- where each method puts its variance
fig3  per-variable, per-level RMSE ratio against the free baseline
fig4  terrain-stratified error and the consistency certificate
fig5  global maps of t2m RMSE and its reduction
"""
import os, sys, json
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, LAT_HR, LON_HR, LAT_LR, LON_LR, area_weights
from orolift.metrics import zonal_spectrum
from orolift.varlist import VARS_69, LEVELS, ATMOS_PREFIXES, SURFACE_VARS
import importlib.util
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "figure.dpi": 160,
                     "savefig.dpi": 160, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
DEV = "cuda"


def coast(ax, lsm, extent=None, lw=0.5, c="k"):
    lo, la = np.meshgrid(LON_HR, LAT_HR)
    ax.contour(lo, la, lsm, levels=[0.5], colors=c, linewidths=lw)
    if extent:
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_xticks([]); ax.set_yticks([])


def main():
    mean, std = load_norm()
    mt = torch.tensor(mean, device=DEV)[:, None, None]
    st = torch.tensor(std, device=DEV)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(DEV)
    sh, sl = static_features(DEV); sh, sl = sh[None], sl[None]
    rs = np.load(f"{DATA}/resid_std.npy")
    model, cfg = ev.load_model("runs/orolift_base/best.pt", rs, DEV)
    lsm = sh[0, 3].cpu().numpy()
    oro_m = (sh[0, 0] * 1000).cpu().numpy()
    ds = HR15("test")

    def run(y):
        x = C.coarsen(y)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            p = model(x, sl, sh).float()
        return x, C.upsample(x), C.project(C.upsample(x), x), p

    # ---------------------------------------------------------------- fig 1
    ti = int(np.where(np.array([str(t) for t in ds.times]) == "2020-01-16 00:00:00")[0][0])
    y = ds[ti][None].to(DEV)
    x, bil, base, pred = run(y)
    iv = VARS_69.index("2m_temperature")
    to_K = lambda a: (a[0, iv] * st[iv] + mt[iv]).cpu().numpy()
    T = {"bilinear + projection": to_K(base), "OroLIFT": to_K(pred),
         "ERA5 1.5° (truth)": to_K(y)}
    x_K = (x[0, iv] * st[iv] + mt[iv]).cpu().numpy()          # on the native 32x64 grid
    EXT = (60, 110, 15, 50)
    vmin, vmax = np.percentile(T["ERA5 1.5° (truth)"], [1, 99])
    m = (LAT_HR >= EXT[2]) & (LAT_HR <= EXT[3]); n = (LON_HR >= EXT[0]) & (LON_HR <= EXT[1])
    sub = lambda a: a[np.ix_(m, n)]
    vmin, vmax = np.percentile(sub(T["ERA5 1.5° (truth)"]), [1, 99])

    fig, ax = plt.subplots(2, 4, figsize=(11, 4.6), constrained_layout=True)
    lo, la = np.meshgrid(LON_HR, LAT_HR)
    loL, laL = np.meshgrid(LON_LR, LAT_LR)
    # draw the input on its own 32x64 grid so the actual coarse resolution is visible
    im = ax[0, 0].pcolormesh(loL, laL, x_K, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="nearest")
    coast(ax[0, 0], lsm, EXT); ax[0, 0].set_title("5.625° input (native grid)")
    for j, (k, v) in enumerate(T.items(), start=1):
        im = ax[0, j].pcolormesh(lo, la, v, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
        coast(ax[0, j], lsm, EXT); ax[0, j].set_title(k)
    fig.colorbar(im, ax=ax[0, :], shrink=0.85, label="2 m temperature (K)", pad=0.01)

    imo = ax[1, 0].pcolormesh(lo, la, oro_m, cmap="terrain", shading="auto")
    coast(ax[1, 0], lsm, EXT); ax[1, 0].set_title("orography (m)")
    fig.colorbar(imo, ax=ax[1, 0], shrink=0.85, pad=0.01)

    eb = T["bilinear + projection"] - T["ERA5 1.5° (truth)"]
    eo = T["OroLIFT"] - T["ERA5 1.5° (truth)"]
    lim = np.percentile(np.abs(sub(eb)), 99)
    for j, (k, e) in enumerate([("bilinear + projection", eb), ("OroLIFT", eo)], start=1):
        ime = ax[1, j].pcolormesh(lo, la, e, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        coast(ax[1, j], lsm, EXT)
        ax[1, j].set_title(f"{k} error\nRMSE = {np.sqrt((sub(e)**2).mean()):.3f} K")
    fig.colorbar(ime, ax=ax[1, 1:3], shrink=0.85, label="error (K)", pad=0.01)

    ax[1, 3].hist(sub(eb).ravel(), bins=60, alpha=0.6, label="bilinear + proj", color="C1")
    ax[1, 3].hist(sub(eo).ravel(), bins=60, alpha=0.6, label="OroLIFT", color="C0")
    ax[1, 3].set_yscale("log"); ax[1, 3].legend(frameon=False, fontsize=7)
    ax[1, 3].set_xlabel("error (K)"); ax[1, 3].set_title("error distribution")
    fig.suptitle(f"2 m temperature, {ds.times[ti]} — Himalaya / Tibetan Plateau", y=1.03)
    fig.savefig(f"{OUT}/fig1_case_t2m.png", bbox_inches="tight"); plt.close(fig)
    print("fig1 done", flush=True)

    # ---------------------------------------------------------------- fig 2
    SPEC = ["2m_temperature", "geopotential_500", "u_component_of_wind_850", "specific_humidity_700"]
    acc = {k: 0 for k in ["truth", "bilinear", "bilinear+proj", "OroLIFT"]}
    dl = torch.utils.data.DataLoader(HR15("test", limit=256), batch_size=16, num_workers=4)
    nb = 0
    for yb in dl:
        yb = yb.to(DEV)
        x, bil, base, pred = run(yb)
        for k, v in [("truth", yb), ("bilinear", bil), ("bilinear+proj", base), ("OroLIFT", pred)]:
            acc[k] = acc[k] + zonal_spectrum(v * st + mt).cpu().numpy()
        nb += 1
    acc = {k: v / nb for k, v in acc.items()}

    fig, ax = plt.subplots(1, 4, figsize=(12, 2.9), constrained_layout=True)
    kk = np.arange(acc["truth"].shape[1])
    style = {"truth": ("k", "-", 1.4), "bilinear": ("C1", "--", 1.0),
             "bilinear+proj": ("C2", "-.", 1.0), "OroLIFT": ("C0", "-", 1.2)}
    for j, v in enumerate(SPEC):
        i = VARS_69.index(v)
        for k, (c, ls, lw) in style.items():
            ax[j].loglog(kk[1:], acc[k][i][1:], color=c, ls=ls, lw=lw, label=k)
        ax[j].axvline(32, color="gray", lw=0.8, ls=":")
        ax[j].text(34, ax[j].get_ylim()[0] * 3, "5.625° Nyquist", fontsize=6, color="gray", rotation=90)
        ax[j].set_title(v); ax[j].set_xlabel("zonal wavenumber")
        if j == 0:
            ax[j].set_ylabel("power"); ax[j].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Zonal power spectra, 60°S–60°N (test 2020, 256 samples)", y=1.06)
    fig.savefig(f"{OUT}/fig2_spectra.png", bbox_inches="tight"); plt.close(fig)
    print("fig2 done", flush=True)

    # ---------------------------------------------------------------- fig 3
    b = json.load(open("runs/eval/baseline_test_synthetic.json"))
    o = json.load(open("runs/eval/orolift50k_test_synthetic.json"))
    ratio = np.array(o["rmse"]) / np.array(b["rmse"])
    M = np.array([[ratio[VARS_69.index(f"{p}_{l}")] for l in LEVELS] for p in ATMOS_PREFIXES])
    S = np.array([[ratio[VARS_69.index(v)] for v in SURFACE_VARS]])

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.6), constrained_layout=True,
                           gridspec_kw={"height_ratios": [5, 1.6]})
    norm = TwoSlopeNorm(vmin=0.25, vcenter=0.65, vmax=1.0)
    im = ax[0].imshow(M, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax[0].set_xticks(range(13), [str(l) for l in LEVELS], fontsize=7)
    ax[0].set_yticks(range(5), ["geopotential", "u wind", "v wind", "temperature", "spec. humidity"])
    ax[0].set_xlabel("pressure level (hPa)")
    for i in range(5):
        for j in range(13):
            ax[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.5)
    im2 = ax[1].imshow(S, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax[1].set_xticks(range(4), ["2m T", "10m u", "10m v", "MSLP"]); ax[1].set_yticks([])
    for j in range(4):
        ax[1].text(j, 0, f"{S[0,j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="RMSE(OroLIFT) / RMSE(bilinear+projection)")
    fig.suptitle("Per-variable skill against the zero-parameter baseline (lower is better)", y=1.05)
    fig.savefig(f"{OUT}/fig3_pervar.png", bbox_inches="tight"); plt.close(fig)
    print("fig3 done", flush=True)

    # ---------------------------------------------------------------- fig 4
    tb, to = np.array(b["terrain"]), np.array(o["terrain"])
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), constrained_layout=True)
    bins = ["0–100 m", "100–400 m", ">400 m"]
    for j, v in enumerate(["2m_temperature", "temperature_1000", "u_component_of_wind_850"]):
        i = VARS_69.index(v); w = 0.36; xx = np.arange(3)
        ax[j].bar(xx - w/2, tb[:, i], w, label="bilinear + projection", color="C1")
        ax[j].bar(xx + w/2, to[:, i], w, label="OroLIFT", color="C0")
        for k in range(3):
            ax[j].text(k + w/2, to[k, i], f"−{100*(1-to[k,i]/tb[k,i]):.0f}%", ha="center",
                       va="bottom", fontsize=6.5)
        ax[j].set_xticks(xx, bins); ax[j].set_title(v)
        ax[j].set_ylabel("RMSE" + (" (K)" if "temp" in v else " (m/s)"))
        if j == 0:
            ax[j].legend(frameon=False, fontsize=7)
    ax[0].set_xlabel("|sub-grid orography|")
    fig.suptitle("Where downscaling pays: error stratified by unresolved topography", y=1.06)
    fig.savefig(f"{OUT}/fig4_terrain.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    vals = [9.0e-2, 6.3e-8, 7.3e-13]
    lbl = ["no projection\n(unconstrained)", "projection\nfp32", "projection\nfp64"]
    ax.bar(range(3), vals, color=["C3", "C0", "C0"])
    ax.set_yscale("log"); ax.set_xticks(range(3), lbl, fontsize=7)
    ax.set_ylabel(r"$\|A\,y - x\|$  (normalized units)")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.7, f"{v:.1e}", ha="center", fontsize=7)
    ax.set_title("Consistency certificate", fontsize=8.5)
    ax.set_ylim(1e-14, 1)
    fig.savefig(f"{OUT}/fig5_certificate.png", bbox_inches="tight"); plt.close(fig)
    print("fig4/5 done", flush=True)

    # ---------------------------------------------------------------- fig 6
    dl = torch.utils.data.DataLoader(HR15("test", limit=256), batch_size=16, num_workers=4)
    se_b = 0; se_o = 0; nb = 0
    for yb in dl:
        yb = yb.to(DEV); x, bil, base, pred = run(yb)
        yy = (yb * st + mt)[:, iv]
        se_b = se_b + (((base * st + mt)[:, iv] - yy) ** 2).sum(0)
        se_o = se_o + (((pred * st + mt)[:, iv] - yy) ** 2).sum(0)
        nb += yb.shape[0]
    rb = np.sqrt((se_b / nb).cpu().numpy()); ro = np.sqrt((se_o / nb).cpu().numpy())
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.4), constrained_layout=True)
    vmax = np.percentile(rb, 99)
    for j, (t, a) in enumerate([("bilinear + projection", rb), ("OroLIFT", ro)]):
        im = ax[j].pcolormesh(lo, la, a, cmap="magma_r", vmin=0, vmax=vmax, shading="auto")
        coast(ax[j], lsm, c="w", lw=0.4)
        ax[j].set_title(f"{t} — 2 m temperature RMSE (K)")
        fig.colorbar(im, ax=ax[j], shrink=0.9, pad=0.01)
    red = 100 * (1 - ro / rb)
    im = ax[2].pcolormesh(lo, la, red, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-20, vcenter=25, vmax=80),
                          shading="auto")
    coast(ax[2], lsm, c="w", lw=0.4); ax[2].set_title("RMSE reduction (%)")
    fig.colorbar(im, ax=ax[2], shrink=0.9, pad=0.01)
    fig.savefig(f"{OUT}/fig6_global_t2m.png", bbox_inches="tight"); plt.close(fig)
    print("fig6 done", flush=True)


if __name__ == "__main__":
    main()
