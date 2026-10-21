"""Canonical 69-variable order (identical to the stormer 1.5deg / 5.625deg configs)."""

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

ATMOS_PREFIXES = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
]

VARS_69 = SURFACE_VARS + [f"{p}_{l}" for p in ATMOS_PREFIXES for l in LEVELS]
assert len(VARS_69) == 69, len(VARS_69)

VAR_INDEX = {v: i for i, v in enumerate(VARS_69)}


def group_of(v):
    """Physical group of a variable name -> used for per-group reporting."""
    for p in ATMOS_PREFIXES:
        if v.startswith(p + "_") and v.rsplit("_", 1)[1].isdigit():
            return p
    return "surface"


def level_of(v):
    tail = v.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None
