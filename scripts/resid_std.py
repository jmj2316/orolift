"""Per-variable std of the free-baseline residual (bilinear + projection).

Used as the output scale of the model and as the loss weight, so that all 69
variables contribute equally: without it the loss is dominated by wind and
humidity, whose residuals are ~10x larger relative to their own std than
geopotential's.
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15
from orolift.grids import SeparableCoarsener, area_weights
from orolift.varlist import VARS_69

dev = "cuda" if torch.cuda.is_available() else "cpu"
C = SeparableCoarsener("conservative", dtype=torch.float32).to(dev)
w = torch.as_tensor(area_weights(), device=dev, dtype=torch.float32)[:, None]
dl = torch.utils.data.DataLoader(HR15("val"), batch_size=16, num_workers=6)
s2 = torch.zeros(69, device=dev, dtype=torch.float64); n = 0
for y in dl:
    y = y.to(dev)
    x = C.coarsen(y)
    r = y - C.project(C.upsample(x), x)
    s2 += (w * r.double() ** 2).mean(dim=(-2, -1)).sum(0)
    n += y.shape[0]
rs = torch.sqrt(s2 / n).cpu().numpy()
np.save("data/resid_std.npy", rs.astype(np.float32))
o = np.argsort(-rs)
print("largest residuals (normalized units):")
for i in o[:6]:
    print(f"   {VARS_69[i]:26s} {rs[i]:.4f}")
print("smallest:")
for i in o[-4:]:
    print(f"   {VARS_69[i]:26s} {rs[i]:.4f}")
print(f"\nmean {rs.mean():.4f}  -> saved data/resid_std.npy")
