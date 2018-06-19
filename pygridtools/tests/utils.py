from pkg_resources import resource_filename
from contextlib import contextmanager
from functools import wraps
import filecmp

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest

import geopandas


def assert_textfiles_equal(baselinefile, outputfile):
    assert filecmp.cmp(baselinefile, outputfile)


def assert_shapefiles_equal(baselinefile, outputfile, atol=0.001):
    expected = geopandas.read_file(baselinefile)
    result = geopandas.read_file(outputfile)
    pdtest.assert_frame_equal(
        result.drop('geometry', axis=1).sort_index(axis='columns'),
        expected.drop('geometry', axis=1).sort_index(axis='columns')
    )
    assert result.geom_almost_equals(expected).all()


def raises(error):
    """Wrapper around pytest.raises to support None."""
    if error:
        return pytest.raises(error)
    else:
        @contextmanager
        def not_raises():
            try:
                yield
            except Exception as e:
                raise e
        return not_raises()


def requires(module, modulename):
    def outer_wrapper(function):
        @wraps(function)
        def inner_wrapper(*args, **kwargs):
            if module is None:
                raise RuntimeError(
                    "{} required for `{}`".format(modulename, function.__name__)
                )
            else:
                return function(*args, **kwargs)
        return inner_wrapper
    return outer_wrapper
