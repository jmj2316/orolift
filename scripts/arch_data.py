"""Real fields and feature maps for the architecture schematic (one test sample)."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener
from orolift.varlist import VARS_69
import importlib.util
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

DEV = "cuda"
mean, std = load_norm()
mt = torch.tensor(mean, device=DEV)[:, None, None]; st = torch.tensor(std, device=DEV)[:, None, None]
C = SeparableCoarsener("conservative", dtype=torch.float32).to(DEV)
sh, sl = static_features(DEV); sh, sl = sh[None], sl[None]
rs = np.load(f"{DATA}/resid_std.npy")
model, cfg = ev.load_model("runs/orolift_base/best.pt", rs, DEV)
ds = HR15("test")
ti = int(np.where(np.array([str(t) for t in ds.times]) == "2020-01-16 00:00:00")[0][0])
y = ds[ti][None].to(DEV)
x = C.coarsen(y)
iv = VARS_69.index("2m_temperature")
K = lambda a: (a[0, iv] * st[iv] + mt[iv]).cpu().numpy()

feats = {}
model.head.register_forward_hook(lambda m, i, o: feats.__setitem__("stem", o.detach()))
model.body[3].register_forward_hook(lambda m, i, o: feats.__setitem__("mid", o.detach()))
model.tail.register_forward_hook(lambda m, i, o: feats.__setitem__("tail", o.detach()))
with torch.no_grad():
    pred = model(x, sl, sh).float()
    up = C.project(C.upsample(x), x)
    direct = model.sample_to_hr(model.to_direct(feats["tail"]))
    raw = model.sample_to_hr(x) + 0  # bilinear lookup alone

def chans(f, idx):
    return np.stack([f[0, i].float().cpu().numpy() for i in idx])

out = dict(
    x_t2m=K(x), up_t2m=K(raw), proj_t2m=K(up), pred_t2m=K(pred), truth_t2m=K(y),
    oro_hr=(sh[0, 0] * 1000).cpu().numpy(), oro_detail=(sh[0, 1] * 1000).cpu().numpy(),
    lsm=sh[0, 3].cpu().numpy(), oro_lr=(sl[0, 0] * 1000).cpu().numpy(),
    stem=chans(feats["stem"], [3, 17, 41]), mid=chans(feats["mid"], [5, 22, 60]),
    tail=chans(feats["tail"], [2, 30, 77]), direct=chans(direct, [1, 9, 20]),
    err_proj=K(up) - K(y), err_pred=K(pred) - K(y),
)
np.savez("figures/arch_inputs.npz", **out)
print({k: v.shape for k, v in out.items()})
