import os
import sys

import nose.tools as nt
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import pandas
import fiona

from pygridtools import io
import testing


class test__outputfile(object):
    def test_basic(self):
        nt.assert_equal(
            io._outputfile('this', 'that.txt'),
            os.path.join('this', 'that.txt')
        )

    def test_withNone(self):
        nt.assert_equal(
            io._outputfile(None, 'that.txt'),
            os.path.join('.', 'that.txt')
        )


class test__check_mode(object):
    @nt.raises(ValueError)
    def test_errors(self):
        io._check_mode('z')

    def test_upper(self):
        nt.assert_equal(io._check_mode('A'), 'a')

    def test_lower(self):
        nt.assert_equal(io._check_mode('w'), 'w')


class test__check_elev_or_mask(object):
    def setup(self):
        self.mainshape = (8, 7)
        self.offset = 2
        self.offsetshape = tuple([s - self.offset for s in self.mainshape])
        self.X = np.zeros(self.mainshape)
        self.Y = np.zeros(self.mainshape)
        self.Yoffset = np.zeros(self.offsetshape)

    @nt.raises(ValueError)
    def test_failNone(self):
        io._check_elev_or_mask(self.X, None, failNone=True)

    @nt.raises(ValueError)
    def test_bad_shape(self):
        io._check_elev_or_mask(self.X, self.Yoffset)

    def test_offset(self):
        other = io._check_elev_or_mask(self.X, self.Yoffset,
                                       offset=self.offset)
        nptest.assert_array_equal(other, self.Yoffset)

    def test_nooffset(self):
        other = io._check_elev_or_mask(self.X, self.Y, offset=0)
        nptest.assert_array_equal(other, self.Y)


class test__check_for_same_masks(object):
    def setup(self):
        from numpy import nan
        self.X = np.array([
            1, 2, 3, nan, nan,   7,
            1, 2, 3, nan, nan,   7,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan,   7,
        ])

        self.Y1 = np.array([
            1, 2, 3, nan, nan,   7,
            1, 2, 3, nan, nan,   7,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan,   7,
        ])

        self.Y2 = np.array([
            1, 2, 3, nan, nan,   7,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan,   7,
        ])

    @nt.raises(ValueError)
    def test_error(self):
        io._check_for_same_masks(self.X, self.Y2)

    def test_baseline(self):
        x, y = io._check_for_same_masks(self.X, self.Y1)
        nptest.assert_array_equal(self.X, x.data)
        nptest.assert_array_equal(self.Y1, y.data)


class test_loadBoundaryFromShapefile(object):
    def setup(self):
        self.shapefile = 'tests/test_data/simple_boundary.shp'
        self.known_df_columns = ['x', 'y', 'beta', 'upperleft',
        					     'reach', 'order']
        self.known_points_in_boundary = 19
        self.test_reach = 1
        self.known_points_in_testreach = 10

    @nptest.dec.skipif(sys.version_info[0] == 3)
    def test_nofilter(self):
        df = io.loadBoundaryFromShapefile(self.shapefile)
        nt.assert_true(isinstance(df, pandas.DataFrame))
        nt.assert_list_equal(df.columns.tolist(), self.known_df_columns)
        nt.assert_equal(df.shape[0], self.known_points_in_boundary)

    @nptest.dec.skipif(sys.version_info[0] == 3)
    def test_filter(self):
        df = io.loadBoundaryFromShapefile(
            self.shapefile,
            filterfxn=lambda r: r['properties']['reach'] == self.test_reach
        )
        nt.assert_equal(df.shape[0], self.known_points_in_testreach)


class test_loadPolygonFromShapefile(object):
    def setup(self):
        self.shpfile = 'tests/test_data/simple_islands.shp'
        self.filter = lambda x: x['properties']['name'] == 'keeper'
        self.known_islands = [
            np.array([
                [10.18915802,  3.71280277], [ 9.34025375,  7.21914648],
                [ 9.34025375,  7.21914648], [10.15224913, 11.98039216],
                [14.1384083 , 11.83275663], [17.6816609 ,  7.47750865],
                [15.94694348,  3.63898501], [10.18915802,  3.71280277]
            ]),
            np.array([
                [ 1.10957324,  3.67589389], [-0.95732411,  7.69896194],
                [-0.95732411,  7.69896194], [ 0.81430219, 10.9100346 ],
                [ 5.98154556, 10.98385236], [ 8.67589389,  7.03460208],
                [ 6.86735871,  3.71280277], [ 1.10957324,  3.67589389]
            ]),
        ]

    @nptest.dec.skipif(sys.version_info[0] == 3)
    def test_baseline(self):
        islands = io.loadPolygonFromShapefile(self.shpfile)
        nt.assert_true(isinstance(islands, list))
        for test, known in zip(islands, self.known_islands):
            nptest.assert_array_almost_equal(test, known)

    @nptest.dec.skipif(sys.version_info[0] == 3)
    def test_filter(self):
        island = io.loadPolygonFromShapefile(self.shpfile, filterfxn=self.filter)
        nt.assert_true(isinstance(island, np.ndarray))
        nptest.assert_array_almost_equal(island, self.known_islands[1])

    @nptest.dec.skipif(sys.version_info[0] == 3)
    def test_filter_nosqueeze(self):
        island = io.loadPolygonFromShapefile(self.shpfile, filterfxn=self.filter,
                                             squeeze=False)
        nt.assert_true(island, list)
        nt.assert_equal(len(island), 1)
        nptest.assert_array_almost_equal(island[0], self.known_islands[1])




def test_dumpGridFile():
    grid = testing.makeSimpleGrid()
    outputfile = 'tests/result_files/grid.out'
    baselinefile = 'tests/baseline_files/grid.out'
    io.dumpGridFiles(grid, 'tests/result_files/grid.out')

    testing.compareTextFiles(outputfile, baselinefile)


class test_savePointShapefile(object):
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
        self.template = 'tests/test_data/schema_template.shp'
        self.outputdir = 'tests/result_files'
        self.baselinedir = 'tests/baseline_files'
        self.river = 'test'

    @nt.raises(ValueError)
    def test_bad_shapes(self):
        io.savePointShapefile(self.x, self.y[:, :1], self.template, 'junk', 'w')

    @nt.raises(ValueError)
    def test_bad_mode(self):
        io.savePointShapefile(self.x, self.y, self.template, 'junk', 'r')

    @nptest.dec.skipif(sys.version_info[0] == 3)
    def test_with_arrays(self):
        fname = 'array_point.shp'
        outfile = os.path.join(self.outputdir, fname)
        basefile = os.path.join(self.baselinedir, fname)
        io.savePointShapefile(self.x, self.y, self.template, outfile,
                              'w', river=self.river)

        testing.compareShapefiles(outfile, basefile)

    @nptest.dec.skipif(sys.version_info[0] == 3)
    def test_with_masks(self):
        fname = 'mask_point.shp'
        outfile = os.path.join(self.outputdir, fname)
        basefile = os.path.join(self.baselinedir, fname)
        io.savePointShapefile(np.ma.MaskedArray(self.x, self.mask),
                              np.ma.MaskedArray(self.y, self.mask),
                              self.template, outfile, 'w', river=self.river)

        testing.compareShapefiles(outfile, basefile)


class test_saveGridShapefile(object):
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
        self.template = 'tests/test_data/schema_template.shp'
        self.outputdir = 'tests/result_files'
        self.baselinedir = 'tests/baseline_files'
        self.river = 'test'
        self.maxDiff=None

    @nt.raises(ValueError)
    def test_bad_mode(self):
        outfile = os.path.join(self.outputdir, 'junk.shp')
        io.saveGridShapefile(self.grid.x, self.grid.y, self.mask,
                             self.template, outfile, mode='junk')

    def test_with_arrays(self):
        fname = 'array_grid.shp'
        outfile = os.path.join(self.outputdir, fname)
        basefile = os.path.join(self.baselinedir, fname)
        io.saveGridShapefile(self.grid.x, self.grid.y, self.mask,
                             self.template, outfile, 'w', river=self.river,
                             elev=None)

        testing.compareShapefiles(basefile, outfile, atol=0.001)


class test_shapefileToDataFrame(object):
    def setup(self):
        pass

    @nt.raises(NotImplementedError)
    def test_placeHolder(self):
        raise NotImplementedError


class test_write_cellinp(object):
    def setup(self):
        self.cells = np.array([
            [9, 9, 9, 9, 9, 9, 0, 0, 0],
            [9, 3, 5, 5, 2, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 5, 5, 9],
            [9, 4, 5, 5, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 0, 0, 0],
        ])
        self.basic_output = 'tests/result_files/cell_basic.inp'
        self.known_basic_output = 'tests/baseline_files/cell_basic.inp'

        self.chunked_output = 'tests/result_files/cell_chunked.inp'
        self.known_chunked_output = 'tests/baseline_files/cell_chunked.inp'

    def test_basic(self):
        io._write_cellinp(self.cells, self.basic_output)
        testing.compareTextFiles(
            self.basic_output,
            self.known_basic_output
        )

    def test_chunked(self):
        io._write_cellinp(self.cells, self.chunked_output, maxcols=5)
        testing.compareTextFiles(
            self.chunked_output,
            self.known_chunked_output
        )


class test__write_gefdc_control_file(object):
    def setup(self):
        self.title
        self.max_i


class test_gridextToShapefile(object):
    def setup(self):
        self.gridextfile = 'tests/test_data/gridext.inp'
        self.template = 'tests/test_data/schema_template.shp'
        self.outputfile = 'tests/result_files/gridext.shp'
        self.baselinefile = 'tests/baseline_files/gridext.shp'
        self.river = 'test'
        self.reach = 1

    def test_basic(self):
        io.gridextToShapefile(self.gridextfile, self.outputfile,
                              self.template, river=self.river)

        testing.compareShapefiles(self.outputfile, self.baselinefile)


    @nt.raises(ValueError)
    def test_bad_input_file(self):
        io.gridextToShapefile('junk', self.outputfile,
                              self.template, river=self.river)

    @nt.raises(ValueError)
    def test_bad_template_file(self):
        io.gridextToShapefile(self.gridextfile, self.outputfile,
                              '/junkie/mcjunk.shp', river=self.river)
