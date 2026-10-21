import json, os, sys
KEY = ["2m_temperature", "mean_sea_level_pressure", "geopotential_500",
       "temperature_850", "u_component_of_wind_850", "specific_humidity_700"]
RUNS = ["cmp_orolift", "cmp_liif", "cmp_edsr", "abl_noproj", "abl_nooro",
        "orolift_ms", "orolift_base"]
print(f"{'run':14s}{'rel':>7s}  " + "".join(f"{k.split('_')[0][:5]:>16s}" for k in KEY))
for r in RUNS:
    p = f"runs/{r}/log.jsonl"
    if not os.path.exists(p):
        continue
    d = json.loads(open(p).read().strip().split("\n")[-1])
    row = "".join(f"{d['rmse'][k]:11.4g}({d['spec'][k]:.2f})" for k in KEY)
    print(f"{r:14s}{d['rel_skill']:7.4f}  {row}")
