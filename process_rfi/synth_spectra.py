#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.time import Time

from process_rfi import PATHS


@dataclass
class SignalChannel:
    index: int
    bandwidth: float
    freq_center: float
    freq_start: float
    freq_stop: float

    def __post_init__(self):
        assert self.index in (*range(1, 9), *range(11, 19))
        # Convert frequencies in GHz to MHz
        self.freq_center *= 1e3
        self.freq_start *= 1e3
        self.freq_stop *= 1e3

    @property
    def is_uplink(self) -> bool:
        return 11 <= self.index <= 18

    @property
    def is_downlink(self) -> bool:
        return 1 <= self.index <= 8


ALL_CHANNELS = {
        chan.index: chan
        for chan in [
                # Frequencies in GHz
                # Downlink channels
                SignalChannel( 1, 250.0, 10.82500, 10.70000, 10.95000),
                SignalChannel( 2, 250.0, 11.07500, 10.95000, 11.20000),
                SignalChannel( 3, 250.0, 11.32500, 11.20000, 11.45000),
                SignalChannel( 4, 250.0, 11.57500, 11.45000, 11.70000),
                SignalChannel( 5, 250.0, 11.82500, 11.70000, 11.95000),
                SignalChannel( 6, 250.0, 12.07500, 11.95000, 12.20000),
                SignalChannel( 7, 250.0, 12.32500, 12.20000, 12.45000),
                SignalChannel( 8, 250.0, 12.57500, 12.45000, 12.70000),
                # Uplink channels
                SignalChannel(11,  62.5, 14.03125, 14.00000, 14.06250),
                SignalChannel(12,  62.5, 14.09375, 14.06250, 14.12500),
                SignalChannel(13,  62.5, 14.15625, 14.12500, 14.18750),
                SignalChannel(14,  62.5, 14.21875, 14.18750, 14.25000),
                SignalChannel(15,  62.5, 14.28125, 14.25000, 14.31250),
                SignalChannel(16,  62.5, 14.34375, 14.31250, 14.37500),
                SignalChannel(17,  62.5, 14.40625, 14.37500, 14.43750),
                SignalChannel(18,  62.5, 14.46875, 14.43750, 14.50000),
        ]
}


def parse_sheet(filen, sheet_name):
    df = pd.read_excel(filen, sheet_name=sheet_name)
    df["eb_ix"] = sheet_name
    df.rename(columns={"unix_ms": "unix_start"}, inplace=True)
    df["unix_start"] /= 1e3  # ms to s
    durations = np.ones(len(df))
    # NOTE This fudges the last transaction to be one second in duration.
    durations[:-1] = df.unix_start[1:].values - df.unix_start[:-1].values
    df["duration"] = durations
    df["unix_end"] = df.eval("unix_start + duration")
    df["mjd_start"] = Time(df.unix_start, format="unix").mjd
    df["mjd_end"] = Time(df.unix_end, format="unix").mjd
    df["mjd_start_s"] = df.mjd_start * u.day.to("s")
    df["mjd_end_s"] = df.mjd_end * u.day.to("s")
    df.loc[df.rx_chan == 0, "rx_chan"] = np.nan
    df.loc[df.rx_chan % 1 != 0, "rx_chan"] = np.nan
    df.loc[df.tx_chan % 1 != 0, "tx_chan"] = np.nan
    df["rx_flo"] = np.nan
    df["rx_fhi"] = np.nan
    df["tx_flo"] = np.nan
    df["tx_fhi"] = np.nan
    for ix in df[df.rx_chan.notnull() | df.tx_chan.notnull()].index:
        row = df.loc[ix]
        if not np.isnan(row.rx_chan):
            chan = ALL_CHANNELS[row.rx_chan]
            df.loc[ix, "rx_flo"] = chan.freq_start
            df.loc[ix, "rx_fhi"] = chan.freq_stop
        if not np.isnan(row.tx_chan):
            chan = ALL_CHANNELS[row.tx_chan]
            df.loc[ix, "tx_flo"] = chan.freq_start
            df.loc[ix, "tx_fhi"] = chan.freq_stop
    return df


def parse_excel_report(filen):
    path = PATHS.data / filen
    assert path.exists()
    with pd.ExcelFile(path) as xl:
        sheet_names = xl.sheet_names
    all_df = []
    for sheet_name in sheet_names:
        if sheet_name == "misc":
            continue
        eb_df = parse_sheet(filen, sheet_name)
        all_df.append(eb_df)
    df = pd.concat(all_df, ignore_index=True)
    return df


class Emitter:
    def __init__(self, df, delta_t=2.0):
        """
        Container-type for received/transmitted data packet timing data.

        Parameters
        ----------
        df : pandas.DataFrame
        delta_t : int
            Minimum time difference to mark as RFI occurring if less than 1.0
            second in duration.
        """
        self.df = df
        self.delta_t = delta_t

    @classmethod
    def from_ods(cls, filen, **kwargs):
        df = parse_excel_report(filen)
        return cls(df, **kwargs)

    @property
    def time_min(self):
        return self.df.mjd_start_s.min()

    @property
    def time_max(self):
        return self.df.mjd_end_s.max()

    def spec_like(self, scan):
        """
        Create a synthetic dynamic spectrum of the same time and frequency
        shape as the scan.

        Parameters
        ----------
        scan : sdmpy.Scan

        Returns
        -------
        data : numpy.ndarray
        extent : list
        """
        freq = scan.freqs().ravel() * u.Hz.to("MHz")
        time = scan.times() * u.day.to("s")
        tmin = time.min()
        tmax = time.max()
        r_time = time - time.min()
        extent = (freq.min(), freq.max(), r_time.min(), r_time.max())
        F, T = np.meshgrid(freq, time)
        data = np.zeros_like(F)
        # Select channels with known transmission and filter by time.
        df = (self.df
                .query("rx_chan.notnull() or tx_chan.notnull()", engine="python")
                .query("mjd_start_s > @tmin and mjd_start_s < @tmax")
        )
        if tmin > self.time_max or tmax < self.time_min or df.shape[0] == 0:
            return data, extent
        # Fill in values in data array to create synthetic dynamic spectrum
        for _, row in df.iterrows():
            chan_cols = [
                    ["rx_chan", "rx_flo", "rx_fhi"],
                    ["tx_chan", "tx_flo", "tx_fhi"],
            ]
            for kind_col, f_lo_col, f_hi_col in chan_cols:
                if pd.notnull(row[kind_col]):
                    cols = [f_lo_col, f_hi_col, "mjd_start_s", "mjd_end_s"]
                    f_lo, f_hi, t_lo, t_hi = row[cols]
                    if (t_hi - t_lo) < 1.0:
                        t_hi = t_lo + self.delta_t
                    data[(F > f_lo) & (F < f_hi) & (T > t_lo) & (T < t_hi)] = 1.0
        return data, extent


