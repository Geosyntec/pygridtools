import os
import sys

import numpy as np
import pandas

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest

from pygridtools import iotools
from pygridtools import testing


class Test__outputfile(object):
    def test_basic(self):
        assert iotools._outputfile('this', 'that.txt') == os.path.join('this', 'that.txt')

    def test_withNone(self):
        assert iotools._outputfile(None, 'that.txt') == os.path.join('.', 'that.txt')


class Test_loadBoundaryFromShapefile(object):
    def setup(self):
        self.shapefile = 'pygridtools/tests/test_data/simple_boundary.shp'
        self.known_df_columns = [
            'x', 'y', 'beta', 'upperleft',
            'reach', 'order'
        ]
        self.known_points_in_boundary = 19
        self.test_reach = 1
        self.known_points_in_testreach = 10

    def test_nofilter(self):
        df = iotools.loadBoundaryFromShapefile(self.shapefile)
        assert (isinstance(df, pandas.DataFrame))
        assert (df.columns.tolist() == self.known_df_columns)
        assert (df.shape[0] == self.known_points_in_boundary)

    def test_filter(self):
        df = iotools.loadBoundaryFromShapefile(
            self.shapefile,
            filterfxn=lambda r: r['properties']['reach'] == self.test_reach
        )
        assert (df.shape[0] == self.known_points_in_testreach)


class Test_loadPolygonFromShapefile(object):
    def setup(self):
        self.shpfile = 'pygridtools/tests/test_data/simple_islands.shp'
        self.filter = lambda x: x['properties']['name'] == 'keeper'
        self.known_islands = [
            np.array([
                [10.18915802,  3.71280277], [ 9.34025375,  7.21914648],
                [ 9.34025375,  7.21914648], [10.15224913, 11.98039216],
                [14.13840830, 11.83275663], [17.68166090,  7.47750865],
                [15.94694348,  3.63898501], [10.18915802,  3.71280277]
            ]),
            np.array([
                [ 1.10957324,  3.67589389], [-0.95732411,  7.69896194],
                [-0.95732411,  7.69896194], [ 0.81430219, 10.91003460],
                [ 5.98154556, 10.98385236], [ 8.67589389,  7.03460208],
                [ 6.86735871,  3.71280277], [ 1.10957324,  3.67589389]
            ]),
        ]

    def test_baseline(self):
        islands = iotools.loadPolygonFromShapefile(self.shpfile)
        assert (isinstance(islands, list))
        for test, known in zip(islands, self.known_islands):
            nptest.assert_array_almost_equal(test, known)

    def test_filter(self):
        island = iotools.loadPolygonFromShapefile(self.shpfile, filterfxn=self.filter)
        assert (isinstance(island, np.ndarray))
        nptest.assert_array_almost_equal(island, self.known_islands[1])

    def test_filter_nosqueeze(self):
        island = iotools.loadPolygonFromShapefile(self.shpfile, filterfxn=self.filter,
                                                  squeeze=False)
        assert isinstance(island, list)
        assert (len(island) == 1)
        nptest.assert_array_almost_equal(island[0], self.known_islands[1])


def test_dumpGridFile():
    grid = testing.makeSimpleGrid()
    outputfile = 'pygridtools/tests/result_files/grid.out'
    iotools.dumpGridFiles(grid, outputfile)

    if sys.platform == 'win32':
        baselinefile = 'pygridtools/tests/baseline_files/grid_win.out'
    else:
        baselinefile = 'pygridtools/tests/baseline_files/grid.out'

    testing.compareTextFiles(outputfile, baselinefile)


class Test_savePointShapefile(object):
    def setup(self):
        self.x = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ])
        self.y = np.array([
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7]
        ])
        self.mask = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])
        self.template = 'pygridtools/tests/test_data/schema_template.shp'
        self.outputdir = 'pygridtools/tests/result_files'
        self.baselinedir = 'pygridtools/tests/baseline_files'
        self.river = 'test'

    def test_bad_shapes(self):
        with pytest.raises(ValueError):
            iotools.savePointShapefile(self.x, self.y[:, :1], self.template, 'junk', 'w')

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            iotools.savePointShapefile(self.x, self.y, self.template, 'junk', 'r')

    def test_with_arrays(self):
        fname = 'array_point.shp'
        outfile = os.path.join(self.outputdir, fname)
        basefile = os.path.join(self.baselinedir, fname)
        iotools.savePointShapefile(self.x, self.y, self.template, outfile,
                                   'w', river=self.river)

        testing.compareShapefiles(outfile, basefile)

    def test_with_masks(self):
        fname = 'mask_point.shp'
        outfile = os.path.join(self.outputdir, fname)
        basefile = os.path.join(self.baselinedir, fname)
        iotools.savePointShapefile(np.ma.MaskedArray(self.x, self.mask),
                                   np.ma.MaskedArray(self.y, self.mask),
                                   self.template, outfile, 'w', river=self.river)

        testing.compareShapefiles(outfile, basefile)


class Test_saveGridShapefile(object):
    def setup(self):
        self.grid = testing.makeSimpleGrid()
        self.mask = np.array([
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ])
        self.template = 'pygridtools/tests/test_data/schema_template.shp'
        self.outputdir = 'pygridtools/tests/result_files'
        self.baselinedir = 'pygridtools/tests/baseline_files'
        self.river = 'test'
        self.maxDiff = None

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            outfile = os.path.join(self.outputdir, 'junk.shp')
            iotools.saveGridShapefile(self.grid.x, self.grid.y, self.mask,
                                      self.template, outfile, mode='junk')

    def test_with_arrays(self):
        fname = 'array_grid.shp'
        outfile = os.path.join(self.outputdir, fname)
        basefile = os.path.join(self.baselinedir, fname)
        iotools.saveGridShapefile(self.grid.x, self.grid.y, self.mask,
                                  self.template, outfile, 'w', river=self.river,
                                  elev=None)

        testing.compareShapefiles(basefile, outfile, atol=0.001)


class Test_shapefileToDataFrame(object):
    def setup(self):
        pass

    def test_placeHolder(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError


class Test_write_cellinp(object):
    def setup(self):
        self.cells = np.array([
            [9, 9, 9, 9, 9, 9, 0, 0, 0],
            [9, 3, 5, 5, 2, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 5, 5, 9],
            [9, 4, 5, 5, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 0, 0, 0],
        ])
        self.basic_output = 'pygridtools/tests/result_files/cell_basic.inp'
        self.known_basic_output = 'pygridtools/tests/baseline_files/cell_basic.inp'

        self.chunked_output = 'pygridtools/tests/result_files/cell_chunked.inp'
        self.known_chunked_output = 'pygridtools/tests/baseline_files/cell_chunked.inp'

    def test_basic(self):
        iotools._write_cellinp(self.cells, self.basic_output)
        testing.compareTextFiles(
            self.basic_output,
            self.known_basic_output
        )

    def test_chunked(self):
        iotools._write_cellinp(self.cells, self.chunked_output, maxcols=5)
        testing.compareTextFiles(
            self.chunked_output,
            self.known_chunked_output
        )


class Test_gridextToShapefile(object):
    def setup(self):
        self.gridextfile = 'pygridtools/tests/test_data/gridext.inp'
        self.template = 'pygridtools/tests/test_data/schema_template.shp'
        self.outputfile = 'pygridtools/tests/result_files/gridext.shp'
        self.baselinefile = 'pygridtools/tests/baseline_files/gridext.shp'
        self.river = 'test'
        self.reach = 1

    def test_basic(self):
        iotools.gridextToShapefile(self.gridextfile, self.outputfile,
                                   self.template, river=self.river)

        testing.compareShapefiles(self.outputfile, self.baselinefile)

    def test_bad_input_file(self):
        with pytest.raises(ValueError):
            iotools.gridextToShapefile('junk__', self.outputfile,
                                       self.template, river=self.river)

    def test_bad_template_file(self):
        with pytest.raises(ValueError):
            iotools.gridextToShapefile(self.gridextfile, self.outputfile,
                                       '/junkie/mcjunk.shp', river=self.river)


def test__write_gefdc_control_file():
    known_filename = 'pygridtools/tests/baseline_files/maingefdc.inp'
    result_filename = 'pygridtools/tests/result_files/maingefdc.inp'
    iotools._write_gefdc_control_file(result_filename, 'Test Input File', 100, 25, 0)
    testing.compareTextFiles(result_filename, known_filename)


def test__write_gridext_file():
    known_filename = 'pygridtools/tests/baseline_files/testgridext.inp'
    result_filename = 'pygridtools/tests/result_files/testgridext.inp'
    df = pandas.DataFrame(np.array([
        [1.25, 3, 4, 3.75],
        [1.75, 4, 4, 3.25],
        [1.25, 4, 5, 3.75],
        [1.75, 5, 5, 3.25],
    ]), columns=['x', 'ii', 'jj', 'y'])
    iotools._write_gridext_file(df, result_filename, icol='ii', jcol='jj',
                                xcol='x', ycol='y')
    testing.compareTextFiles(result_filename, known_filename)


class Test__write_gridout_file(object):
    def setup(self):
        self.known_filename = 'pygridtools/tests/baseline_files/testgrid.out'
        self.result_filename = 'pygridtools/tests/result_files/testgrid.out'

        self.x, self.y = testing.makeSimpleNodes()

    def test_baseline(self):
        iotools._write_gridout_file(self.x, self.y, self.result_filename)
        testing.compareTextFiles(self.result_filename, self.known_filename)

    def test_errors(self):
        with pytest.raises(ValueError):
            iotools._write_gridout_file(self.x, self.y[2:, 2:], 'junk')


class Test_readGridShapefile(object):
    def setup(self):
        self.point_file = "pygridtools/tests/baseline_files/array_point.shp"
        self.cell_file = "pygridtools/tests/baseline_files/array_grid.shp"
        index = pandas.MultiIndex.from_product(
            [[0, 1, 2, 3], [0, 1, 2]],
            names=list('ji')
        )
        x = [1., 2., 3.] * 4
        y = sorted([4., 5., 6., 7.] * 3)
        data = {
            'easting': x,
            'northing': y
        }
        self.known_df = pandas.DataFrame(data=data, index=index)
        self.known_df['elev'] = 0.
        self.known_df['river'] = 'test'
        self.ocols = ['elev', 'river']

    def test_read_pointfile(self):
        result_df = iotools.readGridShapefile(self.point_file, othercols=self.ocols)
        pdtest.assert_frame_equal(result_df, self.known_df)

    def test_read_cellfile(self):
        with pytest.raises(NotImplementedError):
            result_df = iotools.readGridShapefile(self.cell_file)
            pdtest.assert_frame_equal(result_df, self.known_df)
