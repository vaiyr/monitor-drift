"""Shared matplotlib style for preprint figures."""
import matplotlib as mpl

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"       # detection / fresh probes
VERMILLION = "#D55E00" # causal intervention / frozen direction
GREEN = "#009E73"      # cosine / DoM
ORANGE = "#E69F00"     # LR-det
PURPLE = "#CC79A7"     # CV-LR
GREY = "#444444"
LIGHT = "#BBBBBB"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig(fig, stem):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.abspath(os.path.join(here, "..", "figures"))
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"))
