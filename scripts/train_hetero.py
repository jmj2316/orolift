"""Fit the heteroscedastic variance head on top of a frozen OroLIFT."""
import argparse, json, math, os, sys, time, importlib.util
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, area_weights
from orolift.models.hetero import VarHead, gaussian_nll
from orolift.varlist import VARS_69

KEY = ["2m_temperature", "mean_sea_level_pressure", "geopotential_500",
       "temperature_850", "u_component_of_wind_850", "specific_humidity_700"]
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hetero_base")
    ap.add_argument("--det", default="runs/orolift_base/best.pt")
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_limit", type=int, default=192)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.eval_every, a.warmup, a.val_limit = 200, 100, 20, 32
    run = f"runs/{a.name}"; os.makedirs(run, exist_ok=True)
    json.dump(vars(a), open(f"{run}/config.json", "w"), indent=2)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs_np = np.load(f"{DATA}/resid_std.npy")
    rs = torch.tensor(rs_np, device=dev)[None, :, None, None]
    det, _ = ev.load_model(a.det, rs_np, dev)
    for p_ in det.parameters():
        p_.requires_grad_(False)
    det.eval()

    net = VarHead(dim=a.dim, depth=a.depth).to(dev)
    print(f"var head {sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)
    ema = {k: v.detach().clone().float() for k, v in net.named_parameters()}
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    w = torch.as_tensor(area_weights(), device=dev, dtype=torch.float32)[:, None]

    tr = HR15("train"); va = HR15("val", limit=a.val_limit)
    dtr = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, drop_last=True,
                                      num_workers=a.workers, pin_memory=True,
                                      persistent_workers=a.workers > 0, prefetch_factor=4)
    dva = torch.utils.data.DataLoader(va, batch_size=16, num_workers=2)
    log = open(f"{run}/log.jsonl", "a")
    step, t0, best = 0, time.time(), 1e18
    it = iter(dtr)
    while step < a.steps:
        try:
            y = next(it)
        except StopIteration:
            it = iter(dtr); y = next(it)
        y = y.to(dev, non_blocking=True)
        x = C.coarsen(y)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            y_det = det(x, sl, sh).float()
        e = (y - y_det) / rs
        with torch.autocast("cuda", torch.bfloat16):
            ls = net(y_det / rs, sh)
        loss = gaussian_nll(e, ls.float(), w).mean()
        lr = a.lr * (step / max(a.warmup, 1) if step < a.warmup else
                     0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(a.steps - a.warmup, 1))))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, v_ in net.named_parameters():
                ema[k].mul_(a.ema).add_(v_.detach().float(), alpha=1 - a.ema)
        step += 1
        if step % 200 == 0:
            print(f"step {step:6d}  nll {loss.item():.5f}  lr {lr:.2e}  "
                  f"{step/(time.time()-t0):.2f} it/s", flush=True)

        if step % a.eval_every == 0 or step == a.steps:
            pd = dict(net.named_parameters())
            bak = {k: v_.detach().clone() for k, v_ in pd.items()}
            with torch.no_grad():
                for k, v_ in ema.items():
                    pd[k].copy_(v_)
            net.eval()
            nll = 0; z2 = 0; sig = 0; nb = 0
            with torch.no_grad():
                for yv in dva:
                    yv = yv.to(dev); xv = C.coarsen(yv)
                    with torch.autocast("cuda", torch.bfloat16):
                        yd = det(xv, sl, sh).float()
                        lsv = net(yd / rs, sh).float()
                    ev_ = (yv - yd) / rs
                    nll = nll + gaussian_nll(ev_, lsv, w) * yv.shape[0]
                    s = torch.exp(lsv)
                    z2 = z2 + (w * (ev_ / s) ** 2).mean(dim=(-2, -1)).sum(0)
                    sig = sig + (w * s).mean(dim=(-2, -1)).sum(0)
                    nb += yv.shape[0]
            nll, z2, sig = nll / nb, z2 / nb, sig / nb
            net.train()
            with torch.no_grad():
                for k, v_ in bak.items():
                    pd[k].copy_(v_)
            # z2 == 1 means the predicted sigma matches the realized residual
            rec = {"step": step, "nll": float(nll.mean()), "z2": float(z2.mean()),
                   "sigma_mean": float(sig.mean()),
                   "z2_key": {v_: float(z2[VARS_69.index(v_)]) for v_ in KEY},
                   "sigma_key": {v_: float(sig[VARS_69.index(v_)]) for v_ in KEY}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  [eval {step}] nll {float(nll.mean()):.4f}  calib z2 {float(z2.mean()):.3f}  "
                  f"mean sigma {float(sig.mean()):.3f}  " +
                  "  ".join(f"{v_[:10]}:s={sig[VARS_69.index(v_)]:.2f},z2={z2[VARS_69.index(v_)]:.2f}"
                            for v_ in KEY), flush=True)
            if float(nll.mean()) < best:
                best = float(nll.mean())
                torch.save({"ema": ema, "cfg": vars(a), "step": step}, f"{run}/best.pt")
    torch.save({"ema": ema, "cfg": vars(a), "step": step}, f"{run}/last.pt")
    print("done")


if __name__ == "__main__":
    main()
