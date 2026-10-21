"""Paper Table 1, built from the test-split evaluations only."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orolift.varlist import VARS_69

KEY = [("2m_temperature", "t2m (K)", 1),
       ("mean_sea_level_pressure", "MSLP (Pa)", 1),
       ("geopotential_500", "Z500 (m2/s2)", 1),
       ("temperature_850", "T850 (K)", 1),
       ("u_component_of_wind_850", "u850 (m/s)", 1),
       ("specific_humidity_700", "q700 (mg/kg)", 1e6)]
COST = {"orolift_base": (1.83, 5.21),
        "cmp_orolift": (1.83, 5.21), "cmp_liif": (0.93, 10.19),
        "cmp_edsr": (0.67, 19.76), "abl_noproj": (1.77, 5.21),
        "abl_nooro": (1.76, 5.21), "baseline": (0.0, 0.0)}
ORDER = ["baseline", "cmp_liif", "cmp_edsr", "cmp_orolift",
         "abl_noproj", "abl_nooro", "orolift_base"]
NAME = {"baseline": "bilinear + projection", "cmp_liif": "LIIF (25k)",
        "cmp_edsr": "EDSR @ HR (25k)", "cmp_orolift": "OroLIFT (25k)",
        "abl_noproj": "  - projection (25k)", "abl_nooro": "  - orography (25k)",
        "orolift_base": "OroLIFT (50k)"}


def load(tag, inp="synthetic"):
    p = f"runs/eval/{tag}_test_{inp}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    b = load("baseline")
    br = np.array(b["rmse"])
    print("=" * 118)
    print("TABLE 1   test = 2020, 1464 samples, 69 variables, input = A.y (synthetic degradation)")
    print("=" * 118)
    hdr = f"{'model':26s}{'M par':>7s}{'GMAC':>7s}{'rel':>7s}"
    for _, lab, _ in KEY:
        hdr += f"{lab.split()[0]:>13s}"
    print(hdr + f"{'||Ay-x||':>11s}")
    for t in ORDER:
        d = load(t)
        if d is None:
            continue
        r = np.array(d["rmse"]); s = np.array(d["spec"])
        row = f"{NAME[t]:26s}{COST[t][0]:7.2f}{COST[t][1]:7.2f}{(r/br).mean():7.3f}"
        for v, _, sc in KEY:
            i = VARS_69.index(v)
            row += f"{r[i]*sc:9.4g}({s[i]:.2f})".rjust(13)
        print(row + f"{d['cons']:11.1e}")

    print("\n" + "-" * 118)
    print("Physics diagnostics (model vs ERA5 truth) and extremes")
    print(f"{'model':26s}{'hydrostatic':>14s}{'ageostrophic':>14s}{'ACC Z500':>10s}"
          f"{'ACC q700':>10s}{'t2m P99 err':>13s}")
    for t in ORDER:
        d = load(t)
        if d is None:
            continue
        acc = d.get("acc")
        a1 = acc[VARS_69.index("geopotential_500")] if acc else float("nan")
        a2 = acc[VARS_69.index("specific_humidity_700")] if acc else float("nan")
        print(f"{NAME[t]:26s}{np.mean(d['hyd']):14.4g}{np.mean(d['geo']):14.4g}"
              f"{a1:10.4f}{a2:10.4f}{d['q99_key']['2m_temperature']:13.4g}")
    d0 = load("baseline")
    print(f"{'ERA5 truth':26s}{np.mean(d0['hyd_truth']):14.4g}{np.mean(d0['geo_truth']):14.4g}")

    print("\n" + "-" * 118)
    print("t2m RMSE (K) stratified by |sub-grid orography|")
    print(f"{'model':26s}{'0-100 m':>10s}{'100-400 m':>12s}{'>400 m':>10s}")
    i = VARS_69.index("2m_temperature")
    for t in ORDER:
        d = load(t)
        if d is None:
            continue
        te = np.array(d["terrain"])
        print(f"{NAME[t]:26s}{te[0][i]:10.3f}{te[1][i]:12.3f}{te[2][i]:10.3f}")

    print("\n" + "-" * 118)
    print("Operator transfer: stored 5.625-degree files as input instead of A.y")
    for t in ["orolift_base", "orolift_ms"]:
        ds, dr = load(t, "synthetic"), load(t, "real")
        if ds is None or dr is None:
            continue
        rs_, rr = np.array(ds["rmse"]), np.array(dr["rmse"])
        print(f"{NAME[t]:26s} rel {(rs_/br).mean():.3f} -> {(rr/br).mean():.3f}   "
              f"(+{100*((rr/br).mean()/(rs_/br).mean()-1):.1f}%)")


if __name__ == "__main__":
    main()
