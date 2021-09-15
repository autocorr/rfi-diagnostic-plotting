#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

from process_rfi import PATHS

from casatasks import (
        casalog,
        hanningsmooth,
        importasdm,
        listobs,
        rmtables,
)
from casatools import msmetadata
from casaplotms import plotms


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


class ExecutionBlock:
    prefix = "TRFI0004"
    dpi = 300

    def __init__(self, sdm, overwrite=False):
        """
        Parameters
        ----------
        sdm : str
            Filename of the ASDM file for the execution to be processed.
        overwrite : bool, default False
            Overwrite MS files and plots if they exist. If set to ``False``,
            preserve and use existing files.

        Attributes
        ----------
        prefix : str
        """
        self.name = sdm
        self.sdm_path = Path(PATHS.sdm / sdm)
        assert self.sdm_path.exists()
        vis_path = Path(PATHS.vis / sdm)
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

    def plot_crosspower_spectra(self, field_name, scan_id, correlation,
            corr_type="CC"):
        """
        Plot scalar-averaged cross-power spectra for each field, time
        averaging and creating a plot for each scan.
        """
        corr_map = {"AC": "*&&&", "CC": "!*&&&"}
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
                    for corr_type in ("CC", "AC"):
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
                for corr_type, antenna in (("AC", "*&&&"), ("CC", "!*&&&")):
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
    for sdm in sdm_names:
        eb = ExecutionBlock(sdm)
        eb.process()


