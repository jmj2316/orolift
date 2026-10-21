"""Day-of-year climatology from the training split (normalized units), for ACC.

Single pass: accumulate into the sample's own day-of-year bin, then smooth
circularly with a +/- 7 day window.  (The naive "gather each window" version
re-reads the store 15x and would move ~1 TB.)
"""
import os, sys, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.data import HR15, DATA

ds = HR15("train")
doy = np.array([dt.date(*map(int, str(s)[:10].split("-"))).timetuple().tm_yday
                for s in ds.times]) - 1
arr = ds._arr()
acc = np.zeros((366, 69, 121, 240), np.float32)
cnt = np.zeros(366, np.int64)
for i in range(len(ds)):
    acc[doy[i]] += arr[i]
    cnt[doy[i]] += 1
    if (i + 1) % 4000 == 0:
        print(i + 1, flush=True)
W = 7
num = np.zeros_like(acc); den = np.zeros(366, np.float64)
for s in range(-W, W + 1):
    num += np.roll(acc, s, axis=0)
    den += np.roll(cnt, s)
out = (num / den[:, None, None, None].astype(np.float32)).astype(np.float16)
np.save(f"{DATA}/clim_doy.npy", out)
print("saved", out.shape, "counts", cnt.min(), cnt.max())
