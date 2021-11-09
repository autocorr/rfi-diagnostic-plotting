#!/usr/bin/env python3

import os
import copy
import shutil
import multiprocessing
from pathlib import Path
from itertools import compress
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import (gridspec, patheffects, colors)
from matplotlib.ticker import AutoMinorLocator
from matplotlib import patheffects as path_effects

import sdmpy
from astropy import units as u
from scipy.signal import fftconvolve
from scipy.ndimage import median_filter
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
from process_rfi.synth_spectra import Emitter


# Matplotlib configuration settings
plt.rc("text", usetex=True)
plt.rc("font", size=10, family="serif")
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")
plt.ioff()

CMAP = copy.copy(plt.cm.get_cmap("magma"))
CMAP.set_bad("0.5", 1.0)


SCANS_BY_BAND = {
        # (off, off, on)
        # NOTE scan 7 for power-up in Band D
        "a": (1,  6,  9),
        "b": (2, 10, 12),
        "c": (3, 11, 13),
        "d": (4,  5,  7,  8),
}

SDM_FROM_RUN = {
         "0.1": "TRFI0004_sb40126827_1_1.59466.78501478009",
         "0.2": "TRFI0004.sb40134306.eb40140841.59468.47321561343",
         "0.3": "TRFI0004_sb40302025_1_1.59480.94574135417",
         "0.4": "TRFI0004_sb40302025_1_1.59487.62864469907",
         "0.5": "TRFI0004_sb40729015_1_1.59505.731108576394",
         "0.6": "TRFI0004_sb40729015_1_1.59508.7288583912",
         "1.1": "TRFI0004_sb40134306_2_1_20210915_1200.59472.49078583333",
         "1.2": "TRFI0004_sb40134306_2_1_20210915_1545.59472.6525390625",
         "1.3": "TRFI0004_sb40134306_2_1_20210915_1730.59472.725805462964",
         "1.4": "TRFI0004_sb40134306_2_1_20210915_1930.59472.809315266204",
         "2.1": "TRFI0004_sb40134306_2_1_20210916_1200.59473.49042935185",
         "2.2": "TRFI0004_sb40134306_2_1_20210916_1430.59473.60329060185",
         "2.3": "TRFI0004_sb40134306_2_1_20210916_1615.59473.67476782408",
         "2.4": "TRFI0004_sb40134306_2_1_20210916_1845.59473.77818979167",
         "2.5": "TRFI0004_sb40134306_2_1_20210916_2100.59473.87321814815",
         "3.1": "TRFI0004_sb40134306_2_1_20210917_1300.59474.53013979166",
}
RUN_FROM_SDM = {v: k for k, v in SDM_FROM_RUN.items()}

LOCATIONS = {
         "0.1": "N/A",
         "0.2": "N/A",
         "0.3": "N/A",
         "0.4": "N/A",
         "0.5": "N/A",
         "0.6": "N/A",
         "1.1": "VLA Site",
         "1.2": "Route 60",
         "1.3": "Magdalena",
         "1.4": "Clark Field",
         "2.1": "DSOC",
         "2.2": "Alamo Pos. 1",
         "2.3": "Almao Pos. 2",
         "2.4": "Datil",
         "2.5": "Pie Town",
         "3.1": "Unknown",
}
assert all([k in SDM_FROM_RUN for k in LOCATIONS])


@dataclass
class SdmInfo:
    sdm: str
    run_id: str

    @property
    def location(self):
        return LOCATIONS[self.run_id]

    @property
    def run_id(self):
        return RUN_FROM_SDM[self.sdm]

    @property
    def sdm_path(self):
        return PATHS.sdm / self.sdm


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


def split_source_name(source):
    return source.split("=")[-1]


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


class DynamicSpectrum:
    def __init__(self, scan, corr="cross", apply_bandpass=True, uvmax=700,
            mean_filter=None, edge_chan_to_blank=10):
        """
        Parameters
        ----------
        scan : sdmpy.Scan
        corr : str
            Correlation type, valid identifiers ("cross", "auto")
        apply_bandpass : bool; default True
            If set to True, divide out a time-median value per frequency bin
            per polarization per baseline. Note that this may filter time-constant
            signals.
        uvmax : number, None; default 700
            Limit by a maximum *uv*-distance in meters. If `None`, use all values.
            **units**: meters
        mean_filter: tuple, None; default None
            If set to a 2-element tuple (N, M) apply a median filter of N time
            elements and M frequency elements on the data. A shape of (50, 1024)
            will use a window size of 50s (1/6 scan) and 128 MHz (1/7 BW).
        edge_chan_to_blank : int
            Edge to channels to replace with NaNs. Will pass if 0 or None.

        Attributes
        ----------
        data : np.ndarray
        shape : tuple
            (time, baseline/antenna, spw*channel, polarization)
        freq : nd.ndarray
        time : np.ndarray
        baselines : list
            List of antenna pair names for each basline
        baseline_names : list
            Concatenated name string, e.g. "ea01@ea02"
        """
        assert corr in ("cross", "auto")
        self.scan = scan
        self.corr = corr
        self.apply_bandpass = apply_bandpass
        self.uvmax = uvmax
        self.mean_filter = mean_filter
        self.edge_chan_to_blank = edge_chan_to_blank
        self.baselines = scan.baselines
        self.scan_id = int(scan.idx)
        self.source = split_source_name(scan.source)
        # Time and frequency
        freq = scan.freqs().ravel() / 1e6  # Hz to MHz
        time = scan.times()
        self.mjd_s = time
        time = (time - time[0]) * 86400.0  # days to seconds
        self.freq = freq
        self.time = time
        # Read data from BDF, take abs, and reshape for convenience
        # shape -> (t, b/a, s, b, c, p)
        data = np.abs(scan.bdf.get_data(type=corr))
        s = data.shape
        # shape -> (t, b/a, s*b*c, p); note b=bin has length 1
        data = data.reshape(s[0], s[1], s[2]*s[3]*s[4], s[5])
        # shape -> (p, b/a, t, f); for access performance
        data = data.transpose((3, 1, 0, 2)).copy()
        if uvmax is not None and corr == "cross":
            assert uvmax > 0
            # shape.T -> (U, b); note U/uvw coordinate has length 3
            uvw = sdmpy.calib.uvw(
                    scan.startMJD,
                    scan.coordinates,
                    scan.positions,
                    method="astropy",
            ).transpose()
            uvdist = np.sqrt(uvw[0]**2 + uvw[1]**2)
            uvdist_mask = uvdist < uvmax
            if not np.any(uvdist_mask):
                raise ValueError(f"No baselines with uvdist < {uvmax}")
            data = data[:,uvdist_mask,:,:]
            self.baselines = list(compress(self.baselines, uvdist_mask))
        self.baseline_names = [f"{a}@{b}" for a,b in self.baselines]
        # Rudimentary bandpass calibration
        if apply_bandpass:
            time_med = np.nanmedian(data, axis=2, keepdims=True)
            filtered = median_filter(time_med, size=(1, 1, 1, 128))
            data /= filtered
        else:
            # Rudimentary normalization/scaling per baseline/pol
            data_bp = np.nanmedian(data, axis=2, keepdims=True)  # over time
            data_bl = np.nanmean(data_bp, axis=3, keepdims=True)  # over freq
            data_bl[data_bl == 0.0] = np.nan
            data[data == 0.0] = np.nan
            data /= data_bl
        # Apply mean filter
        if mean_filter is not None:
            assert len(mean_filter) == 2
            kernel = np.ones((1, 1, *mean_filter))
            kernel /= kernel.sum()
            data[np.isnan(data)] = 0.0
            flat = fftconvolve(np.ones_like(data), kernel, mode="same", axes=(2, 3))
            fdata = fftconvolve(data, kernel, mode="same", axes=(2, 3))
            data = data - (fdata / flat)
        # Blank out band edges
        if self.edge_chan_to_blank > 0:
            data[:,:,:,:self.edge_chan_to_blank] = np.nan
            data[:,:,:,-self.edge_chan_to_blank:] = np.nan
        # shape -> (polarization, baseline/antenna, time, frequency)
        self.data = data
        self.shape = data.shape
        self.nspec = data.shape[1]

    @property
    def extent(self):
        return [
                self.freq[ 0],
                self.freq[-1],
                self.time[ 0],
                self.time[-1],
        ]

    def get_all_spec(self, pol=0):
        assert pol in (0, 1)
        return self.data[pol]

    def get_spec_by_baseline(self, ant1, ant2):
        assert self.corr == "cross"
        ix = self.baselines.index((ant1, ant2))
        return self.data[:,ix]

    def get_max_spec(self, pol=0):
        spec = self.get_all_spec(pol=pol)
        return np.nanmax(spec, axis=0)  # over baseline/antenna

    def get_mean_spec(self, pol=0):
        spec = self.get_all_spec(pol=pol)
        return np.nanmean(spec, axis=0)  # over baseline/antenna


def get_scan_group(sdm, band="a", **dyna_kwargs):
    """
    Parameters
    ----------
    sdm : sdmpy.SDM
    band : str
    **dyna_kwargs
        Arguments passed to `DynamicSpectrum`
    """
    assert band in SCANS_BY_BAND
    return [
            DynamicSpectrum(sdm.scan(scan_id), **dyna_kwargs)
            for scan_id in SCANS_BY_BAND[band]
    ]


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
        run_id : str
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
        try:
            self.run_id = RUN_FROM_SDM[sdm_name]
        except KeyError:
            raise RuntimeError(f"SDM name not in `RUN_FROM_SDM`: {sdm_name}")
        self.location = LOCATIONS[self.run_id]

    @classmethod
    def from_run_id(cls, run_id, **kwargs):
        sdm_name = SDM_FROM_RUN[run_id]
        return cls(sdm_name, **kwargs)

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
        title = f"{self.run_id} Field={field_name}; Scan={scan_id}; Pol={correlation}; {corr_type}"
        plotfile = (
                PATHS.plot /
                f"D{self.run_id}_{field_name}_{scan_id}_{correlation}_{corr_type}.png"
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
                    title = f"{self.run_id} Field={field_name}; Scan={all_scans_str}; Pol={correlation}; {corr_type}"
                    plotfile = (
                            PATHS.plot /
                            f"D{self.run_id}_{field_name}_avg_{correlation}_{corr_type}.png"
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


class WaterfallPlotter:
    def __init__(self, eb, emitter=None, overwrite=False):
        self.eb = eb
        self.emitter = emitter
        self.overwrite = overwrite
        self.run_id = eb.run_id

    @property
    def keep_existing(self):
        return not self.overwrite

    def add_time_freq_labels(self, ax):
        if ax.is_last_row():
            ax.set_xlabel(r"$\mathrm{Frequency} \ [\mathrm{MHz}]$")
        if ax.is_first_col():
            ax.set_ylabel(r"$\mathrm{Time} \ [\mathrm{s}]$")

    def overplot_emitter_sum(self, ax, scan):
        if self.emitter is None:
            return
        data, extent = self.emitter.summed_spectrum(scan)
        if data.sum() > 0:
            ax.contour(data, origin="lower", extent=extent, levels=[0.5],
                    colors="lime", linewidths=0.5, alpha=0.5)

    def overplot_emitter_by_sat(self, ax, scan, add_legend=True):
        if self.emitter is None:
            return
        extent, F, T = self.emitter.get_extent(scan)
        for i, (data, sat_id) in enumerate(self.emitter.iter_spectra(scan)):
            if data.sum() > 0:
                r, g, b, _ = plt.cm.Set1((i % 9) / 9)
                color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                cs = ax.contour(data, origin="lower", extent=extent,
                        levels=[0.5], colors="black", linewidths=1.5)
                cs = ax.contour(data, origin="lower", extent=extent,
                        levels=[0.5], colors=color, linewidths=0.5)
                if add_legend:
                    ax_pos = (0.02, 0.07 * (1 + i))  # (f, t)
                    txt = ax.annotate(fr"\textbf{{ {sat_id} }}", ax_pos,
                            xycoords="axes fraction", ha="left",
                            va="center_baseline", color=color, fontsize=6)
                    txt.set_path_effects([
                            path_effects.Stroke(linewidth=2, foreground=f"black"),
                            path_effects.Normal(),
                    ])

    def plot_rfi(self, scan, outname=None):
        """
        Plot a dynamic spectrum of the RX/TX transaction data.

        Parameters
        ----------
        scan : sdmpy.Scan
        emitter : Emitter
        outname : str, None
        """
        if self.emitter is None:
            log_post("-- No emitter used, passing.")
            return
        source = split_source_name(scan.source)
        spec, extent = self.emitter.summed_spectrum(scan)
        outname = f"D{self.run_id}_S{scan.idx}_rfi" if outname is None else outname
        if (PATHS.plot/f"{outname}.pdf").exists() and self.keep_existing:
            log_post(f"-- File exists, continuing: {outname}")
            return
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.imshow(spec, cmap=CMAP, origin="upper", extent=extent,
                aspect="auto", vmin=0.0, vmax=1.0)
        self.overplot_emitter_by_sat(ax, scan)
        self.add_time_freq_labels(ax)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(
                f"{self.run_id} Field={source}; Scan={scan.idx}; RFI",
                loc="left",
        )
        plt.tight_layout()
        savefig(f"{outname}.pdf")

    def plot_array_max(self, dyna, vmin=2.0, vmax=3.0, outname=None):
        """
        Plot a waterfall plot for both polarizations by taking the maximum value
        for each antenna/baseline (depending on correlation-type) at every (time,
        frequency) point.

        Parameters
        ----------
        dyna : DynamicSpectrum
        vmin : number
        vmax : number
        outname : str, None
        """
        scan_id = dyna.scan_id
        source = dyna.source
        corr = dyna.corr
        outname = f"D{self.run_id}_S{scan_id}_{corr}" if outname is None else outname
        if (PATHS.plot/f"{outname}.pdf").exists() and self.keep_existing:
            log_post(f"-- File exists, continuing: {outname}")
            return
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True,
                figsize=(4, 4))
        for pol, ax in zip(range(2), axes):
            spec = dyna.get_max_spec(pol=pol)
            ax.imshow(spec, cmap=CMAP, extent=dyna.extent, aspect="auto",
                    vmin=vmin, vmax=vmax)
            self.overplot_emitter_by_sat(ax, dyna.scan)
            ax.set_title(f"P{pol}", loc="right")
            self.add_time_freq_labels(ax)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        axes[0].set_title(
                f"{self.run_id} Field={source}; Scan={scan_id}; {corr.capitalize()}",
                loc="left",
        )
        plt.tight_layout(h_pad=0.3)
        savefig(f"{outname}.pdf")

    def plot_array_max_groups(self, band, dyna_group, vmin=2.0, vmax=3.0,
            outname=None):
        """
        Plot waterfall spectra for groups of spectra at the same band.

        Parameters
        ----------
        band : str
        dyna_group : Iterable(DynamicSpectrum)
        vmin : number
        vmax : number
        outname : str, None
        """
        corr = dyna_group[0].corr
        source = dyna_group[0].source
        scan_ids = ",".join([d.scan.idx for d in dyna_group])
        n_spec = len(dyna_group)
        outname = f"D{self.run_id}_{band.upper()}_{corr}" if outname is None else outname
        if (PATHS.plot/f"{outname}.pdf").exists() and self.keep_existing:
            log_post(f"-- File exists, continuing: {outname}")
            return
        fig = plt.figure(figsize=(8.0, 2.17*n_spec))
        fig.suptitle(
                f"{self.run_id} Field={source}; Band={band.upper()}; Scans={scan_ids}; {corr.capitalize()}",
        )
        outer_grid = gridspec.GridSpec(nrows=1, ncols=2, left=0.10, right=0.95,
                top=0.92, bottom=0.1, wspace=0.15, hspace=0.1)
        for p_ix in range(2):
            inner_grid = gridspec.GridSpecFromSubplotSpec(nrows=n_spec, ncols=1,
                    subplot_spec=outer_grid[p_ix], hspace=0.0)
            for g_ix, dyna in enumerate(reversed(dyna_group)):
                spec = dyna.get_max_spec(pol=p_ix)
                ax = plt.Subplot(fig, inner_grid[g_ix])
                ax.imshow(spec, cmap=CMAP, extent=dyna.extent, aspect="auto",
                        vmin=vmin, vmax=vmax)
                self.overplot_emitter_by_sat(ax, dyna.scan)
                # Axis labeling plumbing
                # FIXME This would be more elegant with the `fig.subfigures`
                #       in matplotlib v3.4
                if g_ix == 0:
                    ax.set_title(f"P{p_ix}", loc="right")
                if g_ix == n_spec-1:
                    ax.set_xlabel(r"$\mathrm{Frequency} \ [\mathrm{MHz}]$")
                if g_ix in range(n_spec-1):
                    ax.set_xticks([])
                if p_ix == 0:
                    ax.set_ylabel(r"$\mathrm{Time} \ [\mathrm{s}]$")
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                fig.add_subplot(ax)
        savefig(f"{outname}.pdf")

    def plot_cross_grid(self, dyna, pol=0, vmin=1.0, vmax=1.6,
            outname=None):
        """
        Plot a grid of individual waterfall plots for each cross-correlation
        per baseline for a given polarization.

        Parameters
        ----------
        dyna : DynamicSpectrum
        pol : int; default 0
        vmin : number
        vmax : number
        outname : str, None
        """
        scan_id = dyna.scan_id
        outname = (
                f"D{self.run_id}_S{scan_id}_P{pol}_all_cross"
                if outname is None else outname
        )
        if (PATHS.plot/f"{outname}.pdf").exists() and self.keep_existing:
            log_post(f"-- File exists, continuing: {outname}")
            return
        spec = dyna.get_all_spec(pol=pol)
        nspec = dyna.nspec
        ncols = 4
        nrows = nspec // ncols
        nrows += 1 if nspec % ncols > 0 else 0
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                figsize=(8.5, 1.55*nrows))
        axiter = iter(axes.flatten())
        for b_ix, ax in zip(range(nspec), axiter):
            ax.imshow(spec[b_ix], cmap=CMAP, extent=dyna.extent, aspect="auto",
                    vmin=vmin, vmax=vmax)
            is_top_left_corner = ax.is_first_row() & ax.is_first_col()
            self.overplot_emitter_by_sat(ax, dyna.scan,
                    add_legend=is_top_left_corner)
            ax.set_title(dyna.baseline_names[b_ix], loc="right",
                    fontdict={"fontsize": 8}, pad=2)
            self.add_time_freq_labels(ax)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        for ax in axiter:
            ax.set_visible(False)
        plt.tight_layout(h_pad=0.3)
        savefig(f"{outname}.pdf")

    def plot_auto_grid(self, dyna, pol=0, vmin=0.995, vmax=1.03,
            outname=None):
        """
        Plot a grid of individual waterfall plots for each autocorrelation per
        antenna for a given polarization.

        Parameters
        ----------
        dyna : DynamicSpectrum
        pol : int, default 0
        vmin : number
        vmax : number
        outname : str, None
        """
        scan_id = dyna.scan_id
        outname = (
                f"D{self.run_id}_S{scan_id}_P{pol}_all_autos"
                if outname is None else outname
        )
        if (PATHS.plot/f"{outname}.pdf").exists() and self.keep_existing:
            log_post(f"-- File exists, continuing: {outname}")
            return
        spec = dyna.get_all_spec(pol=pol)
        fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True,
                figsize=(8.5, 6.3))
        for (ant_ix, ant_name), ax in zip(enumerate(dyna.scan.antennas), axes.flatten()):
            ant_spec = spec[ant_ix]
            ant_spec /= np.nanmedian(ant_spec)
            ant_spec /= np.nanmedian(ant_spec, axis=0, keepdims=True)
            ant_spec /= np.nanmedian(ant_spec, axis=1, keepdims=True)
            ax.imshow(ant_spec, cmap=CMAP, extent=dyna.extent, aspect="auto",
                    vmin=vmin, vmax=vmax)
            self.overplot_emitter_by_sat(ax, dyna.scan)
            ax.set_title(ant_name, loc="right", fontdict={"fontsize": 8}, pad=2)
            self.add_time_freq_labels(ax)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        axes[-1][-2].set_visible(False)  # only 14 antennas with autocorr
        axes[-1][-1].set_visible(False)
        plt.tight_layout()
        savefig(f"{outname}.pdf")

    def plot_all(self, plot_rfi=True, plot_cross=True, plot_band=True,
            plot_auto=True):
        assert self.eb.sdm_path.exists()
        log_post(f"-- Creating all waterfall plots for: {self.eb.name}")
        get_scans = self.eb.sdm.scans  # generator
        # RX/TX plots per scan
        if plot_rfi:
            for scan in get_scans():
                self.plot_rfi(scan)
        # Cross-correlation specific plots
        if plot_cross:
            for scan in get_scans():
                dyna = DynamicSpectrum(scan, corr="cross")
                self.plot_array_max(dyna)
                for pol in (0, 1):
                    self.plot_cross_grid(dyna, pol=pol)
                del dyna
        # Cross-correlation group plots.
        # NOTE This is wasteful because it reads the BDF data again, although
        #      if a sufficient amount of RAM is present this should be in the
        #      filesystem buffer.
        if plot_band:
            for band in list("abcd"):
                group = get_scan_group(self.eb.sdm, band=band)
                self.plot_array_max_groups(band, group)
                del band
        # Auto-correlation specific plots
        if plot_auto:
            for scan in get_scans():
                dyna = DynamicSpectrum(scan, corr="auto")
                self.plot_array_max(dyna)
                for pol in (0, 1):
                    self.plot_auto_grid(dyna, pol=pol)
                del dyna


def get_all_sdm_filenames():
    prefix = ExecutionBlock.prefix
    sdm_names = sorted([
            p.name for p in PATHS.sdm.glob(f"{prefix}*")
    ])
    return sdm_names


def make_waterfalls_from_sdm_name(name):
    eb = ExecutionBlock(name)
    try:
        emitter = Emitter.from_ods("transaction_data.ods")
    except FileNotFoundError:
        emitter = None
    waterfall = WaterfallPlotter(eb, emitter=emitter)
    waterfall.plot_all()


def plot_waterfall_in_parallel():
    sdm_names = get_all_sdm_filenames()
    n_sdm = len(sdm_names)
    with multiprocessing.Pool(n_sdm) as pool:
        pool.map(make_waterfalls_from_sdm_name, sdm_names)


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
        eb = ExecutionBlock(name, overwrite=overwrite)
        eb.process()


def make_all_sdm_plots(overwrite=False):
    """
    Create all plots that only require reading the SDM and not MS creation.
    """
    sdm_names = get_all_sdm_filenames()
    for name in sdm_names:
        eb = ExecutionBlock(name, overwrite=overwrite)
        eb.plot_all_waterfall()


