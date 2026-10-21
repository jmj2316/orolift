"""Train the ker(A)-constrained flow-matching head on top of a frozen OroLIFT."""
import argparse, json, math, os, sys, time, importlib.util
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener, area_weights
from orolift.models.flow import FlowHead, FlowSampler, ker_project
from orolift.varlist import VARS_69
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

KEY = ["2m_temperature", "mean_sea_level_pressure", "geopotential_500",
       "temperature_850", "u_component_of_wind_850", "specific_humidity_700"]


def crps_ensemble(ens, obs, w):
    """Fair CRPS from an M-member ensemble.  ens (B,M,V,H,W), obs (B,V,H,W)."""
    M = ens.shape[1]
    t1 = (ens - obs[:, None]).abs().mean(1)
    d = (ens[:, :, None] - ens[:, None, :]).abs().sum(dim=(1, 2))
    t2 = d / (2 * M * (M - 1))
    return (w * (t1 - t2)).mean(dim=(-2, -1)).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="flow_base")
    ap.add_argument("--det", default="runs/orolift_base/best.pt")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ode_steps", type=int, default=12)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_limit", type=int, default=64)
    ap.add_argument("--sigma_cond", action="store_true",
                    help="condition on a frozen heteroscedastic sigma field")
    ap.add_argument("--var_head", default="runs/het_orolift_base/best.pt")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.eval_every, a.warmup, a.val_limit, a.members, a.ode_steps = 200, 100, 20, 16, 4, 4
    run = f"runs/{a.name}"; os.makedirs(run, exist_ok=True)
    json.dump(vars(a), open(f"{run}/config.json", "w"), indent=2)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    mean, std = load_norm()
    mt = torch.tensor(mean, device=dev)[:, None, None]
    st = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev); sh, sl = sh[None], sl[None]
    rs_np = np.load(f"{DATA}/resid_std.npy")
    rs = torch.tensor(rs_np, device=dev)[None, :, None, None]

    det, _ = ev.load_model(a.det, rs_np, dev)
    for p_ in det.parameters():
        p_.requires_grad_(False)
    det.eval()

    var = None
    if a.sigma_cond:
        from orolift.models.hetero import VarHead
        dv = torch.load(a.var_head, map_location="cpu", weights_only=False)
        var = VarHead(dim=dv["cfg"]["dim"], depth=dv["cfg"]["depth"]).to(dev)
        pv = dict(var.named_parameters())
        with torch.no_grad():
            for k, v_ in dv["ema"].items():
                pv[k].copy_(v_.to(dev))
        var.eval()
        for p_ in var.parameters():
            p_.requires_grad_(False)
        print(f"sigma conditioning from {a.var_head}", flush=True)
    net = FlowHead(dim=a.dim, depth=a.depth, resid_std=rs_np, sigma_cond=a.sigma_cond).to(dev)
    print(f"flow head {sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)
    ema = {k: v.detach().clone().float() for k, v in net.named_parameters()}
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    w = torch.as_tensor(area_weights(), device=dev, dtype=torch.float32)[:, None]

    tr = HR15("train"); va = HR15("val", limit=a.val_limit)
    dtr = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, drop_last=True,
                                      num_workers=a.workers, pin_memory=True,
                                      persistent_workers=a.workers > 0, prefetch_factor=4)
    dva = torch.utils.data.DataLoader(va, batch_size=8, num_workers=2)
    log = open(f"{run}/log.jsonl", "a")
    step, t0, best = 0, time.time(), 1e9
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
        e1 = ker_project(C, (y - y_det) / rs)          # target residual, in ker(A)
        e0 = ker_project(C, torch.randn_like(e1))      # base noise, in ker(A)
        t = torch.rand(y.shape[0], device=dev)
        tt = t[:, None, None, None]
        et = (1 - tt) * e0 + tt * e1
        ls = None
        if var is not None:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                ls = var(y_det / rs, sh).float()
        with torch.autocast("cuda", torch.bfloat16):
            v = net(et, t, y_det / rs, sh, ls)
        loss = (w * (ker_project(C, v.float()) - (e1 - e0)) ** 2).mean()
        lr = a.lr * (step / max(a.warmup, 1) if step < a.warmup else
                     0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(a.steps - a.warmup, 1))))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, v_ in net.named_parameters():
                ema[k].mul_(a.ema).add_(v_.detach().float(), alpha=1 - a.ema)
        step += 1
        if step % 200 == 0:
            print(f"step {step:6d}  loss {loss.item():.5f}  lr {lr:.2e}  gn {gn:.2f}  "
                  f"{step/(time.time()-t0):.2f} it/s", flush=True)

        if step % a.eval_every == 0 or step == a.steps:
            pd = dict(net.named_parameters())
            bak = {k: v_.detach().clone() for k, v_ in pd.items()}
            with torch.no_grad():
                for k, v_ in ema.items():
                    pd[k].copy_(v_)
            net.eval()
            smp = FlowSampler(net, C, det, var)
            cr = sp = sk = cons = 0.0; nb = 0
            for yv in dva:
                yv = yv.to(dev); xv = C.coarsen(yv)
                ens = smp.sample(xv, sl, sh, n_members=a.members, steps=a.ode_steps)
                ensP = ens * st + mt; yP = yv * st + mt
                cr = cr + crps_ensemble(ensP, yP, w) * yv.shape[0]
                m_ = ens.mean(1)
                sk = sk + torch.sqrt((w * ((m_ - yv) * st) ** 2).mean(dim=(-2, -1))).mean(0) * yv.shape[0]
                sp = sp + torch.sqrt((w * (ens.std(1) * st) ** 2).mean(dim=(-2, -1))).mean(0) * yv.shape[0]
                cons += float(C.consistency_error(ens[:, 0], xv).mean()) * yv.shape[0]
                nb += yv.shape[0]
            cr, sp, sk, cons = cr / nb, sp / nb, sk / nb, cons / nb
            ratio = float((sp / sk.clamp_min(1e-12)).mean())
            net.train()
            with torch.no_grad():
                for k, v_ in bak.items():
                    pd[k].copy_(v_)
            rec = {"step": step, "loss": loss.item(), "crps_mean": float(cr.mean()),
                   "spread_skill": ratio, "cons": cons,
                   "crps": {v_: float(cr[VARS_69.index(v_)]) for v_ in KEY}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  [eval {step}] spread/skill {ratio:.3f}  cons {cons:.2e}  " +
                  "  ".join(f"{v_[:12]}={cr[VARS_69.index(v_)]:.4g}" for v_ in KEY), flush=True)
            sc = float(cr.mean())
            if sc < best:
                best = sc
                torch.save({"ema": ema, "cfg": vars(a), "step": step}, f"{run}/best.pt")
    torch.save({"ema": ema, "cfg": vars(a), "step": step}, f"{run}/last.pt")
    print("done")


if __name__ == "__main__":
    main()
