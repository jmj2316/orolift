"""Params / FLOPs / throughput for every architecture, at matched settings.

FLOPs are counted with torch.utils.flop_counter (matmul + conv), which is what the
efficiency claim rests on: OroLIFT's trunk runs on 2048 pixels, the pre-upsampling
baselines run theirs on 29 040.
"""
import os, sys, time, json
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import static_features
from orolift.models.baselines import build_model

CFG = {
    "orolift":  dict(dim=96, depth=8, render_width=128, render_depth=3, direct_dim=32),
    "liif":     dict(dim=64, depth=8),
    "edsr":     dict(dim=64, depth=8),
    "rcan":     dict(dim=64, depth=8),
    "safmn":    dict(dim=64, depth=8),
}

def main():
    dev = "cuda"
    sh, sl = static_features(dev)
    sh, sl = sh[None], sl[None]
    rs = np.load("data/resid_std.npy")
    rows = []
    for arch, cfg in CFG.items():
        m = build_model(arch, resid_std=rs, **cfg).to(dev).eval()
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        x = torch.randn(1, 69, 32, 64, device=dev)
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc, torch.no_grad():
            m(x, sl, sh)
        gflops = fc.get_total_flops() / 2 / 1e9          # MACs
        xb = torch.randn(16, 69, 32, 64, device=dev)
        with torch.no_grad():
            for _ in range(3): m(xb, sl, sh)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10): m(xb, sl, sh)
            torch.cuda.synchronize()
        sps = 160 / (time.time() - t)
        rows.append((arch, n / 1e6, gflops, sps))
        del m; torch.cuda.empty_cache()
    print(f"{'arch':10s}{'params(M)':>12s}{'GMACs/sample':>15s}{'samples/s':>12s}")
    for a, n, g, s in rows:
        print(f"{a:10s}{n:12.2f}{g:15.2f}{s:12.1f}")
    json.dump(rows, open("runs/bench.json", "w"))

if __name__ == "__main__":
    main()
