"""OroLIFT training."""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features
from orolift.grids import SeparableCoarsener, area_weights, cellcentered_grid
from orolift.metrics import lat_weighted_rmse, spectral_ratio
from orolift.models.baselines import build_model
from orolift.varlist import VARS_69

KEY = ["2m_temperature", "mean_sea_level_pressure", "geopotential_500",
       "temperature_850", "u_component_of_wind_850", "specific_humidity_700"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="orolift_base")
    p.add_argument("--arch", default="orolift",
                   choices=["orolift", "liif", "edsr", "rcan", "safmn"])
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--render_width", type=int, default=64)
    p.add_argument("--render_depth", type=int, default=3)
    p.add_argument("--direct_dim", type=int, default=32)
    p.add_argument("--no_projection", action="store_true")
    p.add_argument("--no_orography", action="store_true")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--val_limit", type=int, default=192)
    p.add_argument("--multiscale", nargs="*", type=float, default=None,
                   help="input resolutions in degrees to sample from each step; "
                        "omit for single-scale training at 5.625")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, dl, C, sl, sh, mean_t, std_t, base_rmse):
    model.eval()
    rm, sp, n = 0, 0, 0
    for y in dl:
        y = y.to("cuda", non_blocking=True)
        x = C.coarsen(y)
        with torch.autocast("cuda", torch.bfloat16):
            p = model(x, sl, sh)
        p, yy = p.float() * std_t + mean_t, y * std_t + mean_t
        rm = rm + lat_weighted_rmse(p, yy) * y.shape[0]
        sp = sp + spectral_ratio(p, yy) * y.shape[0]
        n += y.shape[0]
    model.train()
    rm, sp = (rm / n).cpu(), (sp / n).cpu()
    return rm, sp, float((rm / base_rmse).mean())


def main():
    a = get_args()
    if a.smoke:
        a.steps, a.eval_every, a.warmup, a.val_limit, a.train_limit = 200, 100, 20, 64, 2000
    run = f"runs/{a.name}"
    os.makedirs(run, exist_ok=True)
    json.dump(vars(a), open(f"{run}/config.json", "w"), indent=2)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mean, std = load_norm()
    mean_t = torch.tensor(mean, device=dev)[:, None, None]
    std_t = torch.tensor(std, device=dev)[:, None, None]
    C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
    sh, sl = static_features(dev)
    sh, sl = sh[None], sl[None]
    resid_std = np.load("data/resid_std.npy")

    tr = HR15("train", limit=a.train_limit)
    va = HR15("val", limit=a.val_limit)
    dtr = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, drop_last=True,
                                      num_workers=a.workers, pin_memory=True,
                                      persistent_workers=a.workers > 0, prefetch_factor=4)
    dva = torch.utils.data.DataLoader(va, batch_size=16, num_workers=4)
    print(f"train {len(tr)}  val {len(va)}", flush=True)

    model = build_model(a.arch, dim=a.dim, depth=a.depth, render_width=a.render_width,
                        render_depth=a.render_depth, direct_dim=a.direct_dim,
                        resid_std=resid_std, use_projection=not a.no_projection,
                        use_orography=not a.no_orography).to(dev)
    ema = {k: v.detach().clone().float() for k, v in model.named_parameters()}
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params {nparam/1e6:.2f}M", flush=True)

    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))

    w_lat = torch.as_tensor(area_weights(), device=dev, dtype=torch.float32)[:, None]
    inv_rs = torch.tensor(1.0 / resid_std, device=dev)[:, None, None]

    # ---- multi-scale packs -------------------------------------------------
    # Arbitrary-scale decoders only generalize across factors if they are *trained*
    # across factors (LIIF/LMF recipe).  Everything that depends on the input grid
    # -- coarsener, LR statics, (delta, cell-size) queries, residual scale -- is
    # geometry, so it can be precomputed once per resolution and swapped per step.
    packs = None
    if a.multiscale:
        packs = []
        vs = torch.utils.data.DataLoader(HR15("val", limit=192), batch_size=16, num_workers=2)
        for deg in a.multiscale:
            la, lo = cellcentered_grid(deg)
            Ci = SeparableCoarsener("conservative", dtype=torch.float32,
                                    lat_lr=la, lon_lr=lo).to(dev)
            _, sli = static_features(dev, lat_lr=la, lon_lr=lo)
            model.set_lr_grid(la, lo)
            qi, gi = model.query.clone(), model.grid.clone()
            s2 = torch.zeros(69, device=dev, dtype=torch.float64); nn_ = 0
            with torch.no_grad():
                for yv in vs:
                    yv = yv.to(dev); xv = Ci.coarsen(yv)
                    r = yv - Ci.project(Ci.upsample(xv), xv)
                    s2 += (w_lat * r.double() ** 2).mean(dim=(-2, -1)).sum(0); nn_ += yv.shape[0]
            rsi = torch.sqrt(s2 / nn_).float()
            packs.append(dict(C=Ci, sl=sli[None], q=qi, g=gi, rs=rsi,
                              inv=(1.0 / rsi)[:, None, None], deg=deg))
            print(f"  scale {deg:7.4f}deg ({len(la)}x{len(lo)}, {deg/1.5:.2f}x) "
                  f"resid_std mean {float(rsi.mean()):.4f}", flush=True)
        ref = min(range(len(packs)), key=lambda i: abs(packs[i]["deg"] - 5.625))

        def use(i):
            pk = packs[i]
            model.C, model._C64 = pk["C"], None
            model.query, model.grid, model.resid_std = pk["q"], pk["g"], pk["rs"]
            return pk
        use(ref)

    # free baseline, for the relative-skill number
    base = torch.zeros(69, device=dev)
    nb = 0
    with torch.no_grad():
        for y in dva:
            y = y.to(dev); x = C.coarsen(y)
            p = C.project(C.upsample(x), x)
            base += lat_weighted_rmse(p * std_t + mean_t, y * std_t + mean_t) * y.shape[0]
            nb += y.shape[0]
    base = (base / nb).cpu()
    print("baseline(bilinear+proj) key RMSE: " +
          "  ".join(f"{v.split('_')[0][:4]}{VARS_69.index(v)}={base[VARS_69.index(v)]:.4g}" for v in KEY),
          flush=True)

    log = open(f"{run}/log.jsonl", "a")
    best, step, t0 = 1e9, 0, time.time()
    it = iter(dtr)
    while step < a.steps:
        try:
            y = next(it)
        except StopIteration:
            it = iter(dtr); y = next(it)
        y = y.to(dev, non_blocking=True)
        if packs is not None:
            pk = use(np.random.randint(len(packs)))
            Cs, sl_s, inv = pk["C"], pk["sl"], pk["inv"]
        else:
            Cs, sl_s, inv = C, sl, inv_rs
        x = Cs.coarsen(y)
        lr = a.lr * (step / max(a.warmup, 1) if step < a.warmup else
                     0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(a.steps - a.warmup, 1))))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", torch.bfloat16):
            p = model(x, sl_s, sh)
        loss = (w_lat * ((p.float() - y) * inv) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, v in model.named_parameters():
                ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
        step += 1

        if step % 100 == 0:
            print(f"step {step:6d}  loss {loss.item():.5f}  lr {lr:.2e}  "
                  f"gn {gn:.2f}  {step/(time.time()-t0):.2f} it/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            if packs is not None:
                use(ref)
            pd = dict(model.named_parameters())
            bak = {k: v.detach().clone() for k, v in pd.items()}
            with torch.no_grad():
                for k, v in ema.items():
                    pd[k].copy_(v)
            rm, sp, rel = evaluate(model, dva, C, sl, sh, mean_t, std_t, base)
            with torch.no_grad():
                for k, v in bak.items():
                    pd[k].copy_(v)
            rec = {"step": step, "loss": loss.item(), "rel_skill": rel,
                   "rmse": {v: float(rm[VARS_69.index(v)]) for v in KEY},
                   "spec": {v: float(sp[VARS_69.index(v)]) for v in KEY}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  [eval {step}] rel_skill {rel:.4f}  " +
                  "  ".join(f"{v[:12]}={rm[VARS_69.index(v)]:.4g}({sp[VARS_69.index(v)]:.2f})" for v in KEY),
                  flush=True)
            if rel < best:
                best = rel
                torch.save({"ema": ema, "model": model.state_dict(), "cfg": vars(a),
                            "step": step, "rel": rel}, f"{run}/best.pt")
    torch.save({"ema": ema, "model": model.state_dict(), "cfg": vars(a), "step": step},
               f"{run}/last.pt")
    print(f"done, best rel_skill {best:.4f}")


if __name__ == "__main__":
    main()
