#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

import sdmpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import (patheffects, colors)
from matplotlib.ticker import AutoMinorLocator

from casatasks import (
        casalog,
        hanningsmooth,
        importasdm,
        listobs,
        rmtables,
)
from casatools import msmetadata
from casaplotms import plotms

from process_rfi import PATHS


# Matplotlib configuration settings
plt.rc("text", usetex=True)
plt.rc("font", size=10, family="serif")
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")
plt.ioff()


def log_post(msg, priority="INFO"):
    """
    Post a message to the CASA logger, logfile, and stdout/console.

    Parameters
    ----------
    msg : str
        Message to post.
    priority : str, default "INFO"
        Priority level. Includes "INFO", "WARN", "SEVERE".
    """
    print(msg)
    casalog.post(msg, priority, "rfi_pipe")


def add_ext(path, ext):
    return path.parent / f"{path.name}{ext}"


def savefig(outname, dpi=300, relative=True, overwrite=True):
    outpath = PATHS.plot / outname if relative else Path(outname)
    if outpath.exists() and not overwrite:
        print(f"Figure exists, continuing: {outpath}")
    else:
        print(f"Figure saved to: {outpath}")
        plt.savefig(str(outpath), dpi=dpi)
        plt.close("all")


class MetaData:
    def __init__(self, vis):
        self.vis = Path(vis)
        assert self.vis.exists()
        try:
            msmd = msmetadata()
            msmd.open(str(self.vis))
            self.scannumbers = msmd.scannumbers()
            self.fieldnames = msmd.fieldnames()
            self.scansforfields = msmd.scansforfields()
            self.fieldsforscans = msmd.fieldsforscans()
        except RuntimeError:
            raise
        finally:
            msmd.close()
            msmd.done()


class DynamicSpectrum:
    def __init__(self, scan, corr="cross"):
        """
        Parameters
        ----------
        scan : sdmpy.Scan
        corr : str
            Correlation type, valid identifiers ("cross", "auto")

        Attributes
        ----------
        data : np.ndarray
        shape : tuple
        freq : nd.ndarray
        time : np.ndarray
        """
        assert corr in ("cross", "auto")
        self.scan = scan
        self.corr = corr
        # Read data from BDF, take abs, and reshape for convenience
        data = np.abs(scan.bdf.get_data(type=corr))
        s = data.shape
        data = data.reshape((s[0], s[1], s[2]*s[4], s[5]))
        # Rudimentary bandpass and normalization
        data_bp = np.median(data, axis=0, keepdims=True)  # over time
        data_bl = np.mean(data_bp, axis=2, keepdims=True)  # over freq
        data /= data_bl
        self.data = data
        self.shape = data.shape
        # Time and frequency
        freq = scan.freqs().ravel() / 1e6  # MHz
        time = scan.times()
        time = (time - time[0]) * 86400.0  # seconds
        self.freq = freq
        self.time = time

    def get_spec(self, pol=0):
        assert pol in (0, 1)
        spec = self.data.max(axis=1)[:,:,pol]
        extent = [
                self.freq[ 0],
                self.freq[-1],
                self.time[ 0],
                self.time[-1],
        ]
        return spec, extent


class ExecutionBlock:
    prefix = "TRFI0004"
    dpi = 300

    def __init__(self, sdm_name, overwrite=False):
        """
        Parameters
        ----------
        sdm_name : str
            Filename of the ASDM file for the execution to be processed.
        overwrite : bool, default False
            Overwrite MS files and plots if they exist. If set to ``False``,
            preserve and use existing files.

        Attributes
        ----------
        prefix : str
        sdm : sdmpy.SDM
        sdm_path : pathlib.Path
        vis_path : pathlib.Path
        """
        self.name = sdm_name
        self.sdm_path = Path(PATHS.sdm / sdm_name)
        assert self.sdm_path.exists()
        self.sdm = sdmpy.SDM(str(self.sdm_path), use_xsd=False)
        vis_path = Path(PATHS.vis / sdm_name)
        self.vis_path = add_ext(vis_path, ".ms")
        self.hann_path = add_ext(vis_path, ".ms.hann")
        self.overwrite = overwrite

    @property
    def keep_existing(self):
        return not self.overwrite

    def create_ms(self):
        assert self.sdm_path.exists()
        log_post(":: Converting ASDM to MS")
        def call_import():
            importasdm(
                    asdm=str(self.sdm_path),
                    vis=str(self.vis_path),
            )
            listobs(
                    vis=str(self.vis_path),
                    listfile=str(add_ext(self.vis_path, ".listobs.txt")),
                    overwrite=True,
            )
        if not self.vis_path.exists():
            call_import()
        elif self.overwrite:
            rmtables(str(self.vis_path))
            call_import()
        else:
            log_post(f"-- File exists, continuing: {self.vis_path}")

    def hanning_smooth(self):
        assert self.vis_path.exists()
        log_post(":: Applying Hanning smooth")
        def call_hanning():
            hanningsmooth(
                    vis=str(self.vis_path),
                    outputvis=str(self.hann_path),
            )
        if not self.hann_path.exists():
            call_hanning()
        elif self.overwrite:
            rmtables(str(self.hann_path))
            call_hanning()
        else:
            log_post(f"-- File exists, continuing: {self.hann_path}")

    def plot_waterfall(self, scannum, corr="cross", outname=None):
        """
        Parameters
        ----------
        scannum : int
        corr : str, default "cross"
        outname : (str, None)
        """
        outname = f"{self.name}_{scannum}_{corr}" if outname is None else outname
        scan = self.sdm.scan(scannum)
        dyna = DynamicSpectrum(scan, corr=corr)
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4, 4))
        for pol, ax in zip(range(2), axes):
            spec, extent = dyna.get_spec(pol=pol)
            ax.imshow(np.log10(spec), cmap="plasma", extent=extent, aspect="auto")
            ax.set_ylabel(r"$\mathrm{Time} \ [\mathrm{s}]$")
            txt = ax.annotate(f"P{pol}", xy=(0.9, 0.8), xycoords="axes fraction", fontsize=12)
            txt.set_path_effects([patheffects.withStroke(linewidth=4.5, foreground="w")])
        axes[1].set_xlabel(r"$\mathrm{Frequency} \ [\mathrm{MHz}]$")
        axes[0].set_title(f"Field={scan.source}; Scan={scannum}; {corr.capitalize()}")
        plt.tight_layout()
        savefig(f"{outname}.pdf")

    def plot_all_waterfall(self):
        assert self.sdm_path.exists()
        corr_types = ("cross", "auto")
        for scan in self.sdm.scans():
            for corr in corr_types:
                scan_id = int(scan.idx)
                self.plot_waterfall(scan_id, corr=corr)

    def plot_crosspower_spectra(self, field_name, scan_id, correlation,
            corr_type="cross"):
        """
        Plot scalar-averaged cross-power spectra for each field, time
        averaging and creating a plot for each scan.
        """
        corr_map = {"auto": "*&&&", "cross": "!*&&&"}
        assert self.hann_path.exists()
        assert corr_type in corr_map
        antenna = corr_map[corr_type]
        title = f"Field={field_name}; Scan={scan_id}; Pol={correlation}; {corr_type}"
        plotfile = (
                PATHS.plot /
                f"{self.name}_{field_name}_{scan_id}_{correlation}_{corr_type}.png"
        )
        if plotfile.exists():
            if self.keep_existing:
                log_post(f"File exists, continuing: {plotfile}")
                return
            else:
                log_post(f"Removing file: {plotfile}")
                os.remove(str(plotfile))
        log_post(f"-- Generating plot for: '{title}'")
        plotms(
                vis=str(self.hann_path),
                xaxis="frequency",
                yaxis="amp",
                showgui=False,
                # Data selection
                selectdata=True,
                antenna=antenna,
                scan=str(scan_id),
                correlation=correlation,
                # Data averaging
                averagedata=True,
                scalar=True,
                avgtime="99999",
                avgbaseline=True,
                # Figure properties
                plotfile=str(plotfile),
                dpi=self.dpi,
                width=4*self.dpi,
                height=3*self.dpi,
                highres=True,
                # Draw style
                customsymbol=True,
                symbolshape="circle",
                symbolcolor="black",
                symbolsize=2,
                title=title,
                showmajorgrid=True,
                majorstyle="dash",
        )

    def plot_all_crosspower_spectra_per_scan(self):
        assert self.hann_path.exists()
        meta = MetaData(self.hann_path)
        for field_id in meta.fieldsforscans:
            field_name = meta.fieldnames[field_id]
            for scan_id in meta.scansforfields[str(field_id)]:
                for correlation in ("LL", "RR"):
                    for corr_type in ("cross", "auto"):
                        self.plot_crosspower_spectra(
                                field_name,
                                scan_id,
                                correlation,
                                corr_type=corr_type,
                        )

    def plot_all_crosspower_spectra_per_field(self):
        assert self.hann_path.exists()
        meta = MetaData(self.hann_path)
        for field_id in meta.fieldsforscans:
            field_name = meta.fieldnames[field_id]
            scansforfields = meta.scansforfields[str(field_id)]
            all_scans_str = ",".join(str(n) for n in scansforfields)
            for correlation in ("LL", "RR"):
                for corr_type, antenna in (("auto", "*&&&"), ("cross", "!*&&&")):
                    title = f"Field={field_name}; Scan={all_scans_str}; Pol={correlation}; {corr_type}"
                    plotfile = (
                            PATHS.plot /
                            f"{self.name}_{field_name}_avg_{correlation}_{corr_type}.png"
                    )
                    if plotfile.exists():
                        if self.keep_existing:
                            log_post(f"File exists, continuing: {plotfile}")
                            continue
                        else:
                            log_post(f"Removing file: {plotfile}")
                            os.remove(str(plotfile))
                    log_post(f"-- Generating plot for: '{title}'")
                    plotms(
                            vis=str(self.hann_path),
                            xaxis="frequency",
                            yaxis="amp",
                            showgui=False,
                            # Data selection
                            selectdata=True,
                            antenna=antenna,
                            scan=all_scans_str,
                            correlation=correlation,
                            # Data averaging
                            averagedata=True,
                            scalar=True,
                            avgtime="99999",
                            avgscan=True,
                            avgbaseline=True,
                            # Figure properties
                            plotfile=str(plotfile),
                            dpi=self.dpi,
                            width=4*self.dpi,
                            height=3*self.dpi,
                            highres=True,
                            # Draw style
                            customsymbol=True,
                            symbolshape="circle",
                            symbolcolor="black",
                            symbolsize=2,
                            title=title,
                            showmajorgrid=True,
                            majorstyle="dash",
                    )

    def process(self):
        self.create_ms()
        self.hanning_smooth()
        self.plot_all_crosspower_spectra_per_scan()
        self.plot_all_crosspower_spectra_per_field()


def get_all_sdm_filenames():
    prefix = ExecutionBlock.prefix
    sdm_names = sorted([
            p.name for p in PATHS.sdm.glob(f"{prefix}*")
    ])
    return sdm_names


def process_all_executions(overwrite=False):
    """
    Retrieve all ASDM files set by the ``VIS_DIR`` directory and create plots
    for each.

    Parameters
    ----------
    overwrite : bool, default False
        Overwrite files? If set to True will remove and overwrite files. If set
        to False, will continue and leave files in place. When set to False the
        function should be safe to run simply whenever new data is added.
    """
    sdm_names = get_all_sdm_filenames()
    for name in sdm_names:
        eb = ExecutionBlock(name)
        eb.process()


