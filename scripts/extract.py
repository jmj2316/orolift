"""Extract the 69 ERA5 variables from the 1.5-degree HDF5 store into a
z-scored float16 memmap, plus static fields and normalization constants.

Splits follow the 5.625-degree store exactly (so the operator-mismatch
experiment is apples-to-apples):
    train 1979-2018,  val 2019,  test 2020.

Storage is z-scored so that float16 relative precision (~5e-4) costs ~0.05% of
each field's std -- negligible against the ~3-10% signal a downscaler must
recover -- while cutting the dataset in half.
"""
import argparse, os, sys, time
import numpy as np, h5py
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.varlist import VARS_69

SRC = os.environ["OROLIFT_SRC_1P5"]          # per-time-step HDF5 files of ERA5 at 1.5 degrees
OUT = os.environ.get("OROLIFT_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
NLAT, NLON, NV = 121, 240, 69

SPLITS = {                       # (source subdir, years, stride over the 1460/yr 6-hourly index)
    "train": ("train", range(1979, 2019), 3),   # stride 3 cycles all four synoptic hours
    "val":   ("train", range(2019, 2020), 3),
    "test":  ("val",   range(2020, 2021), 1),
}


def file_list(split):
    sub, years, stride = SPLITS[split]
    out = []
    for y in years:
        names = sorted(f for f in os.listdir(f"{SRC}/{sub}") if f.startswith(f"{y}_"))
        out += [f"{SRC}/{sub}/{n}" for n in names[::stride]]
    return out


def _read(fp):
    with h5py.File(fp, "r") as f:
        g = f["input"]
        a = np.empty((NV, NLAT, NLON), np.float32)
        for i, v in enumerate(VARS_69):
            a[i] = g[v][()].T
        t = g["time"][()]
    return a, (t.decode() if isinstance(t, bytes) else str(t))


def _worker(args):
    idx, fp, path, mean, std = args
    a, t = _read(fp)
    mm = np.memmap(path, np.float16, "r+", shape=(_worker.N, NV, NLAT, NLON))
    mm[idx] = ((a - mean[:, None, None]) / std[:, None, None]).astype(np.float16)
    del mm
    return idx, t


def _init(n):
    _worker.N = n


def compute_stats(nsample=400, seed=0):
    files = file_list("train")
    rng = np.random.default_rng(seed)
    pick = [files[i] for i in rng.choice(len(files), nsample, replace=False)]
    s1 = np.zeros(NV, np.float64); s2 = np.zeros(NV, np.float64); n = 0
    with Pool(16) as p:
        for a, _ in p.imap_unordered(_read, pick, chunksize=2):
            s1 += a.reshape(NV, -1).sum(1); s2 += (a.astype(np.float64).reshape(NV, -1) ** 2).sum(1)
            n += NLAT * NLON
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-30))
    return mean.astype(np.float64), std.astype(np.float64)


def extract_static():
    with h5py.File(file_list("val")[0], "r") as f:
        g = f["input"]
        st = {k: g[k][()].T.astype(np.float32) for k in
              ["geopotential_at_surface", "land_sea_mask"]}
    np.savez(f"{OUT}/static_hr.npz", **st)
    print("static:", {k: (v.shape, float(v.min()), float(v.max())) for k, v in st.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    npz = f"{OUT}/norm.npz"
    if not os.path.exists(npz):
        print("computing normalization constants ...", flush=True)
        mean, std = compute_stats()
        np.savez(npz, mean=mean, std=std, vars=np.array(VARS_69))
        for v, m, s in list(zip(VARS_69, mean, std))[:4]:
            print(f"   {v:26s} mean={m:12.4g} std={s:12.4g}")
    d = np.load(npz, allow_pickle=True)
    mean, std = d["mean"], d["std"]

    if not os.path.exists(f"{OUT}/static_hr.npz"):
        extract_static()

    for split in args.splits:
        files = file_list(split)
        N = len(files)
        path = f"{OUT}/hr15_{split}.f16"
        gb = N * NV * NLAT * NLON * 2 / 1e9
        print(f"\n[{split}] {N} samples -> {path}  ({gb:.1f} GB)", flush=True)
        np.memmap(path, np.float16, "w+", shape=(N, NV, NLAT, NLON)).flush()
        times = [None] * N
        t0 = time.time()
        with Pool(args.workers, initializer=_init, initargs=(N,)) as p:
            jobs = [(i, f, path, mean, std) for i, f in enumerate(files)]
            for k, (i, t) in enumerate(p.imap_unordered(_worker, jobs, chunksize=4)):
                times[i] = t
                if (k + 1) % 500 == 0 or k + 1 == N:
                    el = time.time() - t0
                    print(f"   {k+1}/{N}  {el:6.1f}s  eta {el/(k+1)*(N-k-1):6.1f}s", flush=True)
        np.save(f"{OUT}/times_{split}.npy", np.array(times))
        assert all(t is not None for t in times)
    print("\ndone")


if __name__ == "__main__":
    main()
