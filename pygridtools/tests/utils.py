from pkg_resources import resource_filename
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import filecmp


try:
    import pytest
except ImportError:
    pytest = None

import numpy.testing as nptest
import pandas.util.testing as pdtest

import geopandas


def assert_textfiles_equal(baselinefile, outputfile):
    expectedtext = Path(baselinefile).open('r').read()
    resulttext = Path(outputfile).open('r').read()
    assert filecmp.cmp(baselinefile, outputfile)


def assert_gis_files_equal(baselinefile, outputfile, atol=0.001):
    expected = geopandas.read_file(baselinefile)
    result = geopandas.read_file(outputfile)
    assert_gdfs_equal(expected, result)


def assert_gdfs_equal(expected_gdf, result_gdf):
    pdtest.assert_frame_equal(
        expected_gdf.drop('geometry', axis=1).sort_index(axis='columns'),
        result_gdf.drop('geometry', axis=1).sort_index(axis='columns')
    )
    assert expected_gdf.geom_almost_equals(result_gdf).all()
