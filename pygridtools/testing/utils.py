import numpy as np
from numpy import nan
import pandas
import fiona
import geopandas

import numpy.testing as nptest
import pandas.util.testing as pdtest


def compareTextFiles(baselinefile, outputfile):
    with open(outputfile) as output:
        results = output.read()

    with open(baselinefile) as baseline:
        expected = baseline.read()

    assert (results == expected)


def compareShapefiles(baselinefile, outputfile, atol=0.001):
    expected = geopandas.read_file(baselinefile)
    result = geopandas.read_file(outputfile)
    pdtest.assert_frame_equal(
        result.drop('geometry', axis=1).sort_index(axis='columns'),
        expected.drop('geometry', axis=1).sort_index(axis='columns')
    )
    assert result.geom_almost_equals(expected).all()
