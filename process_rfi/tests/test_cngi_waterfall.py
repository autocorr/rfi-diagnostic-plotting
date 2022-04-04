#!/usr/bin/env python3

from pathlib import Path

import pytest

from process_rfi.cngi_waterfall import *


DATA_ROOT = "/lustre/aoc/sciops/bsvoboda/starlink_rfi/data"
TEST_DATE = "032922"
TEST_MS_NAME = MS_FILES_BY_DATE[TEST_DATE]
VIS_FILEN = os.path.join(DATA_ROOT, f"pipe_{TEST_DATE}", TEST_MS_NAME)


def test_check_existing():
    for date, ms_name in MS_FILES_BY_DATE.items():
        path = Path(DATA_ROOT) / f"pipe_{date}" / ms_name
        assert path.exists()


def test_get_listobs_text():
    text = get_listobs_text(VIS_FILEN)
    assert text is not None
    assert len(text) > 0


def test_read_ms_data():
    assert read_ms_data(VIS_FILEN, band="X") is not None
    assert read_ms_data(VIS_FILEN, band="U") is not None


def test_mxds():
    mxds = read_ms_from_pipe(TEST_DATE)
    assert mxds is not None
    concat_xds = concat_by_spw(mxds)
    assert concat_xds is not None
    dmag_xds = calc_max_cross(concat_xds)
    assert dmag_xds is not None
    assert "DMAG" in dmag_xds


