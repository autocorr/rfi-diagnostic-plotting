#!/usr/bin/env python3
"""
Process RFI
-----------
Top-level module for processing RFI sweep data in X and Ku Bands.

Authors
-------
Brian Svoboda

Copyright 2021 Brian Svoboda under the MIT License.
"""

import os
import sys
import copy
import warnings

from matplotlib import pyplot as plt

IS_PYTHON3 = sys.version_info >= (3, 0, 0)
if not IS_PYTHON3:
    raise ImportError("Must be run under CASA v6 & Python 3.")
else:
    from pathlib import Path
    from configparser import ConfigParser

try:
    casalog.version
    IS_RUNNING_WITHIN_CASA = True
except NameError:
    IS_RUNNING_WITHIN_CASA = False


CONFIG_FILEN = "process.cfg"
cfg_parser = ConfigParser()
if not cfg_parser.read(CONFIG_FILEN):
    warnings.warn(f"File '{CONFIG_FILEN}' not found; falling back to CWD.")
    root = Path(os.getcwd())
    cfg_parser["Paths"] = {
            "ROOT_DIR": str(root),
            "DATA_DIR": str(root/"data"),
            "PLOT_DIR": str(root/"data"/"plots"),
            "SDM_DIR":  str(root/"data"/"sdm_data"),
            "VIS_DIR":  str(root/"data"/"vis_data"),
    }


# Matplotlib configuration settings
plt.rc("text", usetex=True)
plt.rc("font", size=10, family="serif")
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")
plt.ioff()

CMAP = copy.copy(plt.cm.get_cmap("magma"))
CMAP.set_bad("0.5", 1.0)


class ProjPaths:
    def __init__(self, root, data, plot, sdm, vis):
        """
        Absolute paths for data and product files.
        """
        self.root = root
        self.data = data
        self.plot = plot
        self.sdm = sdm
        self.vis = vis
        if not plot.exists():
            plot.mkdir()
        if not vis.exists():
            vis.mkdir()


PATHS = ProjPaths(
        Path(cfg_parser.get("Paths", "ROOT_DIR")),
        Path(cfg_parser.get("Paths", "DATA_DIR")),
        Path(cfg_parser.get("Paths", "PLOT_DIR")),
        Path(cfg_parser.get("Paths", "SDM_DIR")),
        Path(cfg_parser.get("Paths", "VIS_DIR")),
)


def scan_data_to_xarray(scan, corr="cross"):
    import xarray
    if corr == "cross":
        array_dim = "baseline"
        array_coord = [f"{a1}@{a2}" for a1, a2 in scan.baselines]
    elif corr == "auto":
        array_dim = "antenna"
        array_coord = scan.antennas
    else:
        raise ValueError(f"Invalid corr={corr}")
    # shape -> (t, b/a, s, b, c, p); note b/a depends on corr
    data = scan.bdf.get_data(type=corr)
    s = data.shape
    # shape -> (t, b/a, s*b*c, p); note b=bin has length 1
    data = data.reshape(s[0], s[1], s[2]*s[3]*s[4], s[5])
    # Get axis label information
    times = scan.times()
    times = (times - times[0]) * u.day.to("s")
    freqs = scan.freqs().ravel() / 1e6  # MHz
    pols = scan.bdf.spws[0].pols(type=corr)
    xarr = xarray.DataArray(
            data,
            dims=(
                    "time",
                    array_dim,
                    "frequency",
                    "polarization",
            ),
            coords={
                    "time": times,
                    array_dim: array_coord,
                    "frequency": freqs,
                    "polarization": pols,
            },
            attrs=dict(
                    time_unit="sec",
                    frequency_unit="MHz",
                    startMJD=times[0]*u.day.to("s"),
            ),
    )
    return xarr


def savefig(outname, dpi=300, relative=True, overwrite=True):
    outpath = PATHS.plot / outname if relative else Path(outname)
    if outpath.exists() and not overwrite:
        print(f"Figure exists, continuing: {outpath}")
    else:
        print(f"Figure saved to: {outpath}")
        plt.savefig(str(outpath), dpi=dpi)
        plt.close("all")


