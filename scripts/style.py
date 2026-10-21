"""Shared figure style: one font stack, bold lowercase panel tags, one palette for the methods."""
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight", "axes.titleweight": "normal",
    "axes.grid": False, "legend.frameon": False, "lines.linewidth": 1.3,
})

METHODS = ["bilinear", "bilinear+proj", "liif", "edsr", "orolift"]
LABEL = {"bilinear": "Bilinear", "bilinear+proj": "Bilinear + proj.", "liif": "LIIF",
         "edsr": "EDSR (output grid)", "orolift": "OroLIFT", "truth": "ERA5 1.5°", "x": "5.625° input"}
PAL = {"bilinear": "#B7C9E8", "bilinear+proj": "#7F9FD3", "liif": "#3F6BBE", "edsr": "#1D2B55",
       "orolift": "#D64550", "truth": "0.15", "x": "0.5"}
ACCENT = PAL["orolift"]


def tag(ax, s, dx=-0.02, dy=1.03, fs=13):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=fs, fontweight="bold", ha="right", va="bottom")


def note(ax, s, fs=8, loc="bl"):
    x, ha = (0.03, "left") if loc.endswith("l") else (0.97, "right")
    y, va = (0.04, "bottom") if loc.startswith("b") else (0.96, "top")
    ax.text(x, y, s, transform=ax.transAxes, fontsize=fs, ha=ha, va=va, zorder=10,
            bbox=dict(fc="white", ec="0.5", lw=0.5, pad=1.6, alpha=0.9))
