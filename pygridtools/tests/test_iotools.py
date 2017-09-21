import os
import sys
from pkg_resources import resource_filename
import tempfile

import numpy as np
import pandas
from shapely import geometry
import geopandas

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest

from pygridtools import iotools
from pygridtools import testing


@pytest.mark.parametrize(('folder', 'expected'), [
    ('this', os.path.join('this', 'that.txt')),
    (None, os.path.join('.', 'that.txt'))
])
def test__outputfile(folder, expected):
    assert iotools._outputfile(folder, 'that.txt') == expected


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
        np.array([
            [10.18915802,  3.71280277], [9.34025375,  7.21914648],
            [9.34025375,  7.21914648], [10.15224913, 11.98039216],
            [14.13840830, 11.83275663], [17.68166090,  7.47750865],
            [15.94694348,  3.63898501], [10.18915802,  3.71280277]
        ]),
        np.array([
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
def test_write_points(usemasks, fname):
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])
    mask = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool)
    if usemasks:
        x = np.ma.masked_array(x, mask)
        y = np.ma.masked_array(y, mask)

    template = resource_filename('pygridtools.tests.test_data', 'schema_template.shp')
    baselinedir = resource_filename('pygridtools.tests', 'baseline_files')
    river = 'test'
    with tempfile.TemporaryDirectory() as outputdir:
        outfile = os.path.join(outputdir, fname)
        basefile = os.path.join(baselinedir, fname)
        gdf = iotools.write_points(x, y, template, outfile, river=river)
        testing.compareShapefiles(outfile, basefile)
        assert isinstance(gdf, geopandas.GeoDataFrame)


@pytest.mark.parametrize(('usemasks', 'fname'), [
    (False, 'array_grid.shp'),
    # (True, 'mask_grid.shp'),
])
def test_write_cells(usemasks, fname, simple_grid):
    if usemasks:
        mask = np.array([
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

    template = resource_filename('pygridtools.tests.test_data', 'schema_template.shp')
    baselinedir = resource_filename('pygridtools.tests', 'baseline_files')
    river = 'test'
    with tempfile.TemporaryDirectory() as outputdir:
        outfile = os.path.join(outputdir, fname)
        basefile = os.path.join(baselinedir, fname)
        gdf = iotools.write_cells(simple_grid.x, simple_grid.y, mask, template,
                                  outfile, river=river)
        testing.compareShapefiles(basefile, outfile)
        assert isinstance(gdf, geopandas.GeoDataFrame)


@pytest.mark.parametrize(('maxcols', 'knownfile'), [
    (150, resource_filename('pygridtools.tests.baseline_files', 'cell_basic.inp')),
    (5, resource_filename('pygridtools.tests.baseline_files', 'cell_chunked.inp'))
])
def test_write_cellinp(maxcols, knownfile):
    cells = np.array([
        [9, 9, 9, 9, 9, 9, 0, 0, 0],
        [9, 3, 5, 5, 2, 9, 9, 9, 9],
        [9, 5, 5, 5, 5, 5, 5, 5, 9],
        [9, 5, 5, 5, 5, 5, 5, 5, 9],
        [9, 4, 5, 5, 1, 9, 9, 9, 9],
        [9, 9, 9, 9, 9, 9, 0, 0, 0],
    ])
    with tempfile.TemporaryDirectory() as outputdir:
        outfile = os.path.join(outputdir, 'cell.inp')
        iotools._write_cellinp(cells, outfile, maxcols=maxcols)
        testing.compareTextFiles(outfile, knownfile)


def test_convert_gridext_to_shp():
    gridextfile = resource_filename('pygridtools.tests.test_data', 'gridext.inp')
    template = resource_filename('pygridtools.tests.test_data', 'schema_template.shp')
    baselinefile = resource_filename('pygridtools.tests.baseline_files', 'gridext.shp')
    river = 'test'
    reach = 1
    with tempfile.TemporaryDirectory() as outputdir:
        outputfile = os.path.join(outputdir, 'gridext.shp')
        iotools.convert_gridext_to_shp(gridextfile, outputfile, template, river=river)
        testing.compareShapefiles(baselinefile, outputfile)


def test__write_gefdc_control_file():
    with tempfile.TemporaryDirectory() as outputdir:
        result_filename = os.path.join(outputdir, 'maingefdc.inp')
        known_filename = resource_filename('pygridtools.tests.baseline_files', 'maingefdc.inp')
        iotools._write_gefdc_control_file(result_filename, 'Test Input File', 100, 25, 0)
        testing.compareTextFiles(result_filename, known_filename)


def test__write_gridext_file():
    with tempfile.TemporaryDirectory() as outputdir:
        known_filename = resource_filename('pygridtools.tests.baseline_files', 'testgridext.inp')
        result_filename = os.path.join(outputdir, 'testgridext.inp')
        df = pandas.DataFrame(np.array([
            [1.25, 3, 4, 3.75],
            [1.75, 4, 4, 3.25],
            [1.25, 4, 5, 3.75],
            [1.75, 5, 5, 3.25],
        ]), columns=['x', 'ii', 'jj', 'y'])
        iotools._write_gridext_file(df, result_filename, icol='ii', jcol='jj',
                                    xcol='x', ycol='y')
        testing.compareTextFiles(result_filename, known_filename)


def test__write_gridout_file(simple_nodes):
    known_filename = resource_filename('pygridtools.tests.baseline_files', 'testgrid.out')
    x, y = simple_nodes

    with tempfile.TemporaryDirectory() as outdir:
        result_filename = os.path.join(outdir, 'testgrid.out')

        iotools._write_gridout_file(x, y, result_filename)
        testing.compareTextFiles(result_filename, known_filename)


def test_read_grid():
    pntfile = resource_filename('pygridtools.tests.baseline_files', 'array_point.shp')
    cellfile = resource_filename('pygridtools.tests.baseline_files', 'array_grid.shp')
    result_df = iotools.read_grid(pntfile, othercols=['elev', 'river'])
    known_df = pandas.DataFrame(
        data={'easting': [1., 2., 3.] * 4, 'northing': sorted([4., 5., 6., 7.] * 3)},
        index=pandas.MultiIndex.from_product([[0, 1, 2, 3], [0, 1, 2]], names=list('ji'))
    ).assign(elev=0.0).assign(river='test')

    pdtest.assert_frame_equal(result_df, known_df)

    with pytest.raises(NotImplementedError):
        result_df = iotools.read_grid(cellfile)
