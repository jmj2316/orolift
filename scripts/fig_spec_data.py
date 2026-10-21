"""Zonal power spectra (all 69 variables) for truth and five methods, 256 test samples -> figures/fig_spec.npz"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
from orolift.data import HR15, load_norm, static_features, DATA
from orolift.grids import SeparableCoarsener
from orolift.metrics import zonal_spectrum
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

DEV = "cuda"
CK = {"liif": "runs/cmp_liif/best.pt", "edsr": "runs/cmp_edsr/best.pt", "orolift": "runs/cmp_orolift/best.pt"}
mean, std = load_norm()
mt = torch.tensor(mean, device=DEV)[:, None, None]; st = torch.tensor(std, device=DEV)[:, None, None]
C = SeparableCoarsener("conservative", dtype=torch.float32).to(DEV)
sh, sl = static_features(DEV); sh, sl = sh[None], sl[None]
rs = np.load(f"{DATA}/resid_std.npy")
models = {k: ev.load_model(p, rs, DEV)[0] for k, p in CK.items()}
names = ["truth", "bilinear", "bilinear+proj", "liif", "edsr", "orolift"]
acc = {k: 0 for k in names}; nb = 0
for y in torch.utils.data.DataLoader(HR15("test", limit=256), batch_size=16, num_workers=4):
    y = y.to(DEV); x = C.coarsen(y); up = C.upsample(x)
    out = {"truth": y, "bilinear": up, "bilinear+proj": C.project(up, x)}
    for k, m in models.items():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out[k] = m(x, sl, sh).float()
    for k in names:
        acc[k] = acc[k] + zonal_spectrum(out[k] * st + mt).cpu().numpy()
    nb += 1
np.savez("figures/fig_spec.npz", names=np.array(names), spec=np.stack([acc[k] / nb for k in names]))
print("done", nb)
