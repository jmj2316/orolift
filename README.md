# OroLIFT: exact consistency and error decomposition for learned downscaling

Code for the manuscript *Exact consistency and error decomposition for learned
downscaling of atmospheric fields* (Minjong Cheon, submitted to Computers & Geosciences).

The repository does three things.

1. **Exact consistency projection.** For a separable coarsening operator
   `A = A_lat (x) A_lon` between regular latitude-longitude grids, the orthogonal projection
   onto `{y : A y = x}` is closed-form, has no parameters and works at a non-integer factor
   (here 5.625 -> 1.5 degrees, 3.75x) with spherical area weights (`orolift/grids.py`).
2. **Error decomposition.** The squared error of any estimate splits exactly into a part fixed
   by the coarse input (range space) and a part a downscaler can change (null space)
   (`scripts/decomp.py`).
3. **OroLIFT.** A downscaler whose trunk runs on the coarse grid and whose coordinate-queried
   decoder produces the 1.5-degree field (1.83 M parameters, 5.21 GMAC), with EDSR- and
   LIIF-style baselines, a flow-matching head confined to the null space of `A`, and the
   scripts behind every number and figure of the paper.

## Quick test (no data, CPU, a few seconds)

```bash
pip install -r requirements.txt
python tests/quick_test.py
```

The test builds the operator, projects synthetic fields, and checks: consistency to round-off
(fp64 and fp32), idempotence, the exact range/null-space split, the RMSE gain
`1 - sqrt(1 - s)` of projecting an interpolant, and a forward pass of the full network
(random weights) whose output satisfies `A y = x` to about `1e-7`. Expected output:

```
[ok] projection is consistent (fp64): relative residual 8.6e-13
[ok] projection is consistent (fp32): relative residual 7.3e-08
[ok] projection is idempotent: max change 4.0e-12
[ok] error = range-space + null-space: relative residual 6.0e-12
[ok] range-space part is fixed by the input: ...
[ok] RMSE gain = 1 - sqrt(1 - s): ...
[ok] parameter count: 1.83 M
[ok] output shape: (2, 69, 121, 240)
[ok] network output is consistent with its input: |A y - x| = 8.7e-08 (rms)
[ok] without the projection it is not: |A y - x| = 4.8e-01 (rms)

ALL PASSED
```

## Using the projection in your own model

```python
import torch
from orolift.grids import SeparableCoarsener

C = SeparableCoarsener("conservative", dtype=torch.float32)   # 121x240 -> 32x64 (A)
x = C.coarsen(y_true)                    # coarse input, shape (..., 32, 64)
y_hat = my_network(x)                    # any 1.5-degree estimate, shape (..., 121, 240)
y_hat = C.project(y_hat, x)              # now C.coarsen(y_hat) == x to fp32 round-off
```

The operator pins its own arithmetic to true fp32 (no TF32, no bfloat16 autocast); see
`exact_matmul` in `orolift/grids.py`. Other grids are obtained by passing `lat_lr`, `lon_lr`.

## Layout

```
orolift/grids.py        separable coarsener, exact projection, area weights
orolift/data.py         dataset over the float16 store, static high-resolution predictors
orolift/metrics.py      latitude-weighted RMSE/ACC, zonal spectra, quantiles, terrain-stratified RMSE
orolift/models/         orolift.py (OroLIFT), baselines.py (EDSR, LIIF), flow.py, hetero.py
scripts/                training, evaluation, and one script per table/figure (see below)
tests/quick_test.py     the quick test above
data/                   norm.npz, resid_std.npy, static_hr.npz (small files the code needs)
```

## Data

ERA5 (Hersbach et al., 2020) at 1.5 and 5.625 degrees, 69 variables (four surface fields and
five upper-air fields on 13 pressure levels), stored as one HDF5 file per time step in the
layout of the Stormer data pipeline: a group `input` holding one dataset per variable
(stored as longitude x latitude) and a `time` entry; files are named `<year>_<index>.h5` and
sit in `train/` and `val/` sub-directories. Training uses 1979-2018 (every third 6-hourly
step), validation 2019 and testing 2020.

```bash
export OROLIFT_SRC_1P5=/path/to/era5_1.5deg_h5       # train/ and val/ with the 1.5-degree files
export OROLIFT_SRC_5P625=/path/to/era5_5.625deg_h5   # only for the independent-archive check
export OROLIFT_DATA=$PWD/data                        # default; small files live here
export OROLIFT_BIG=/path/with/space                  # optional: where the large float16 stores go
python scripts/extract.py        # builds hr15_{train,val,test}.f16 (about 78, 2 and 6 GB)
python scripts/clim.py           # day-of-year climatology for the anomaly scores
```

## Training and evaluation of the models in the paper

One 48 GB GPU is enough.

```bash
# 25,000-step comparison and ablations (Table 1)
python scripts/train.py --name cmp_orolift --arch orolift --dim 96 --render_width 128 --bs 32 --steps 25000 --eval_every 2500
python scripts/train.py --name cmp_edsr    --arch edsr    --dim 64 --bs 32 --steps 25000 --eval_every 2500
python scripts/train.py --name cmp_liif    --arch liif    --dim 64 --bs 32 --steps 25000 --eval_every 2500
python scripts/train.py --name abl_noproj  --arch orolift --dim 96 --render_width 128 --bs 32 --steps 25000 --no_projection
python scripts/train.py --name abl_nooro   --arch orolift --dim 96 --render_width 128 --bs 32 --steps 25000 --no_orography
# 50,000-step model used for the cyclone gallery and the ensembles
python scripts/train.py --name orolift_base --arch orolift --dim 96 --render_width 128 --bs 32 --steps 50000

python scripts/resid_std.py                                   # output scale (already in data/)
python scripts/evaluate.py --tag baseline                     # bilinear + projection
python scripts/evaluate.py --ckpt runs/cmp_orolift/best.pt --tag cmp_orolift   # likewise for the others
python scripts/table1.py
```

Trained checkpoints are attached to the GitHub release `v1.0.0` as individual files
(`orolift_base.pt`, `cmp_orolift.pt`, `cmp_edsr.pt`, `cmp_liif.pt`, `abl_noproj.pt`,
`abl_nooro.pt`, `hetero_base.pt`, `flow_base.pt`, `flow_v2.pt`, `flow_v2_nocond.pt`). Save each
as `runs/<name>/best.pt`, for example

```bash
mkdir -p runs/orolift_base
curl -L -o runs/orolift_base/best.pt \
  https://github.com/jmj2316/orolift/releases/download/v1.0.0/orolift_base.pt
```

## Which script produces what

| Paper item | Scripts |
|---|---|
| Consistency certificate, zonal spectra | `fig_spec_data.py`, `fig_spec_cert.py` |
| Main comparison table, error maps, terrain classes, case studies | `evaluate.py`, `table1.py`, `fig_data.py`, `fig_maps.py`, `fig_terrain_cases.py`, `figures.py` |
| Parameter and multiply-accumulate counts | `bench.py` |
| Geostrophic balance of the outputs | `balance_probe.py` |
| Error decomposition, persistence stress test | `decomp.py`, `fig_decomp.py` |
| Tropical cyclones (selection, 25-storm comparison, gallery, attribution) | `typhoon.py`, `typhoon_cmp.py`, `fig_tc_cmp.py`, `typhoon_gallery.py`, `typhoon_why.py` |
| Ensembles confined to the null space | `train_hetero.py`, `train_flow.py`, `eval_flow.py`, `fig_flow.py` |
| Spread-skill referee | `sigma_referee.py`, `compare_sigma.py`, `fig_flowabl.py` |
| Irreducible fraction versus capacity and data | `irreducible_map.py`, `irreducible_report.py`, `floor_report.py`, `fig_floor.py` |
| Schematics | `fig_arch.py`, `fig_overview.py` |

Scripts run from the repository root and write to `runs/` and `figures/`.

## Citation

If you use the code, please cite the manuscript above (reference to be updated on publication).

## License

MIT, see `LICENSE`. ERA5 is distributed under the Copernicus license; the small files in
`data/` (normalization constants, orography and land-sea mask) are derived from it.
