"""Quick test: runs on a CPU in under a minute and needs no downloaded data.

It checks the statements the paper rests on, on synthetic smooth fields:
  1. the projection returns fields that coarsen back to the input (fp64 and fp32);
  2. the projection is idempotent;
  3. the squared error splits exactly into a range-space and a null-space part,
     and the range-space part equals the input-defined quantity |X - A yhat|^2 in the
     metric (A W^-1 A^T)^-1;
  4. projecting a bilinear interpolant lowers the RMSE by 1 - sqrt(1 - s), where s is
     the range-space share of its squared error;
  5. the OroLIFT network (with random weights) has 1.83 M parameters, and its output
     is consistent with its input to float32 round-off, while the same network without
     the projection is not.

Run from the repository root:   python tests/quick_test.py
"""
import os, sys, warnings
import numpy as np, torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from orolift.grids import SeparableCoarsener, area_weights
from orolift.models.orolift import OroLIFT
from orolift.data import static_features

torch.manual_seed(0)
rng = np.random.default_rng(0)
NLAT, NLON = 121, 240
fails = []


def check(name, ok, detail):
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


def smooth_field(power=-1.5):
    """Random field with a power-law spectrum on the 121 x 240 grid, unit variance."""
    z = rng.standard_normal((NLAT, NLON))
    ky = np.fft.fftfreq(NLAT)[:, None] * NLAT
    kx = np.fft.fftfreq(NLON)[None, :] * NLON
    f = np.fft.ifft2(np.fft.fft2(z) * (1.0 + ky ** 2 + kx ** 2) ** power).real
    return (f - f.mean()) / f.std()


C64 = SeparableCoarsener("conservative", dtype=torch.float64)
w = torch.tensor(area_weights(), dtype=torch.float64)[:, None]          # latitude weights, mean 1
wsum = lambda e: float((w * e ** 2).sum())

y = torch.tensor(np.stack([smooth_field() for _ in range(4)]))           # truth, 4 channels
x = C64.coarsen(y)                                                       # coarse input, 32 x 64
yhat = C64.upsample(x) + 0.3 * torch.tensor(np.stack([smooth_field(-0.7) for _ in range(4)]))

# 1. consistency ------------------------------------------------------------------------
ystar = C64.project(yhat, x)
r64 = float(torch.linalg.norm(C64.coarsen(ystar) - x) / torch.linalg.norm(x))
check("projection is consistent (fp64)", r64 < 1e-11, f"relative residual {r64:.1e}")
C32 = SeparableCoarsener("conservative", dtype=torch.float32)
y32, x32 = yhat.float(), C32.coarsen(y.float())
r32 = float(torch.linalg.norm(C32.coarsen(C32.project(y32, x32)) - x32) / torch.linalg.norm(x32))
check("projection is consistent (fp32)", r32 < 1e-5, f"relative residual {r32:.1e}")

# 2. idempotence ------------------------------------------------------------------------
d = float((C64.project(ystar, x) - ystar).abs().max())
check("projection is idempotent", d < 1e-11, f"max change {d:.1e}")

# 3. exact split into range and null space ----------------------------------------------
tot = wsum(yhat - y)
rng_part = wsum(yhat - ystar)
null_part = wsum(ystar - y)
res = abs(tot - rng_part - null_part) / tot
check("error = range-space + null-space", res < 1e-10, f"relative residual {res:.1e}")
R = x - C64.coarsen(yhat)
Mi = torch.einsum("ij,cjk,kl->cil", C64.Mi_lat, R, C64.Mi_lon)
from_input = float((R * Mi).sum())
check("range-space part is fixed by the input", abs(from_input - rng_part) / rng_part < 1e-9,
      f"{rng_part:.6g} vs |X - A yhat|^2 in the (A W^-1 A^T)^-1 metric {from_input:.6g}")

# 4. gain of the projection on a bilinear interpolant ------------------------------------
yb = C64.upsample(x)
s = wsum(yb - C64.project(yb, x)) / wsum(yb - y)
gain = 1.0 - (wsum(C64.project(yb, x) - y) / wsum(yb - y)) ** 0.5
check("RMSE gain = 1 - sqrt(1 - s)", abs(gain - (1 - (1 - s) ** 0.5)) < 1e-9,
      f"range share s = {s:.3f}, gain = {100 * gain:.1f} %")

# 5. the network ------------------------------------------------------------------------
sh, sl = static_features("cpu")
sh, sl = sh[None], sl[None]
net = OroLIFT(render_width=128, resid_std=np.full(69, 0.1, np.float32)).eval()
n_par = sum(p.numel() for p in net.parameters())
check("parameter count", 1.80e6 < n_par < 1.86e6, f"{n_par / 1e6:.2f} M")
for p in net.render_out.parameters():             # random weights, so the residual is not zero
    torch.nn.init.normal_(p, std=0.05)
xin = torch.randn(2, 69, 32, 64)
with torch.no_grad():
    out = net(xin, sl, sh)
    net.use_projection = False
    raw = net(xin, sl, sh)
check("output shape", tuple(out.shape) == (2, 69, 121, 240), str(tuple(out.shape)))
res_p = float(torch.sqrt(((C32.coarsen(out) - xin) ** 2).mean()))
res_n = float(torch.sqrt(((C32.coarsen(raw) - xin) ** 2).mean()))
check("network output is consistent with its input", res_p < 1e-5, f"|A y - x| = {res_p:.1e} (rms)")
check("without the projection it is not", res_n > 1e-3, f"|A y - x| = {res_n:.1e} (rms)")

print("\nALL PASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
