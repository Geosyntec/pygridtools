import os
import sys
from pkg_resources import resource_filename
import tempfile

import numpy
import pandas
from shapely import geometry
import geopandas

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest

from pygridtools import iotools
from . import utils


@pytest.mark.parametrize(('filterfxn', 'points_in_boundary'), [
    (None, 19),
])
def test_read_boundary(filterfxn, points_in_boundary):
    shapefile = resource_filename('pygridtools.tests.test_data', 'simple_boundary.shp')
    df = iotools.read_boundary(shapefile, filterfxn=filterfxn)

    df_columns = ['x', 'y', 'beta', 'upperleft', 'reach', 'order', 'geometry']
    assert (isinstance(df, geopandas.GeoDataFrame))
    assert (df.columns.tolist() == df_columns)
    assert (df.shape[0] == points_in_boundary)


@pytest.mark.parametrize('as_gdf', [False, True])
def test_read_polygons(as_gdf):
    shapefile = resource_filename('pygridtools.tests.test_data', 'simple_islands.shp')
    known_islands = [
        numpy.array([
            [10.18915802,  3.71280277], [9.34025375,  7.21914648],
            [9.34025375,  7.21914648], [10.15224913, 11.98039216],
            [14.13840830, 11.83275663], [17.68166090,  7.47750865],
            [15.94694348,  3.63898501], [10.18915802,  3.71280277]
        ]),
        numpy.array([
            [1.10957324,  3.67589389], [-0.95732411,  7.69896194],
            [-0.95732411,  7.69896194], [0.81430219, 10.91003460],
            [5.98154556, 10.98385236], [8.67589389,  7.03460208],
            [6.86735871,  3.71280277], [1.10957324,  3.67589389]
        ]),
    ]
    islands = iotools.read_polygons(shapefile, as_gdf=as_gdf)
    if as_gdf:
        expected = geopandas.GeoDataFrame({
            'id': [2, 1],
            'name': ['loser', 'keeper']
        }, geometry=list(map(geometry.Polygon, known_islands)))
        pdtest.assert_frame_equal(
            islands.drop('geometry', axis='columns'),
            expected.drop('geometry', axis='columns')
        )
        assert islands.geom_almost_equals(expected).all()
    else:
        for res, exp in zip(islands, known_islands):
            nptest.assert_array_almost_equal(res, exp)


@pytest.mark.parametrize(('usemasks', 'fname'), [
    (False, 'array_point.shp'),
    (True, 'mask_point.shp'),
])
def test_write_points(usemasks, fname, example_crs):
    x = numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = numpy.array([[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])
    mask = numpy.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool)
    if usemasks:
        x = numpy.ma.masked_array(x, mask)
        y = numpy.ma.masked_array(y, mask)

    baselinedir = resource_filename('pygridtools.tests', 'baseline_files')
    river = 'test'
    with tempfile.TemporaryDirectory() as outputdir:
        outfile = os.path.join(outputdir, fname)
        basefile = os.path.join(baselinedir, fname)
        gdf = iotools.write_points(x, y, example_crs, outfile, river=river)
        utils.assert_shapefiles_equal(outfile, basefile)
        assert isinstance(gdf, geopandas.GeoDataFrame)


@pytest.mark.parametrize(('usemasks', 'fname'), [
    (False, 'array_grid.shp'),
    # (True, 'mask_grid.shp'),
])
def test_write_cells(usemasks, fname, simple_grid, example_crs):
    if usemasks:
        mask = numpy.array([
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ])
    else:
        mask = None

    baselinedir = resource_filename('pygridtools.tests', 'baseline_files')
    river = 'test'
    with tempfile.TemporaryDirectory() as outputdir:
        outfile = os.path.join(outputdir, fname)
        basefile = os.path.join(baselinedir, fname)
        gdf = iotools.write_cells(simple_grid.x, simple_grid.y, mask, example_crs,
                                  outfile, river=river)
        utils.assert_shapefiles_equal(basefile, outfile)
        assert isinstance(gdf, geopandas.GeoDataFrame)


def test_read_grid():
    pntfile = resource_filename('pygridtools.tests.baseline_files', 'array_point.shp')
    cellfile = resource_filename('pygridtools.tests.baseline_files', 'array_grid.shp')
    result_df = iotools.read_grid(pntfile, othercols=['elev', 'river'])
    known_df = pandas.DataFrame(
        data={'easting': [1., 2., 3.] * 4, 'northing': sorted([4., 5., 6., 7.] * 3)},
        index=pandas.MultiIndex.from_product([[2, 3, 4, 5], [2, 3, 4]], names=['jj', 'ii'])
    ).assign(elev=0.0).assign(river='test').reset_index().set_index(['ii', 'jj']).sort_index()

    pdtest.assert_frame_equal(result_df, known_df)

    with utils.raises(NotImplementedError):
        result_df = iotools.read_grid(cellfile)
