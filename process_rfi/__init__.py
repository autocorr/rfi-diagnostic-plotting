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
import warnings

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
    cfg_parser.set("Paths", "ROOT_DIR", root)
    cfg_parser.set("Paths", "DATA_DIR", root/"data")
    cfg_parser.set("Paths", "PLOT_DIR", root/"data/plots")
    cfg_parser.set("Paths", "SDM_DIR",  root/"data/sdm_data")
    cfg_parser.set("Paths", "VIS_DIR",  root/"data/vis_data")


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


