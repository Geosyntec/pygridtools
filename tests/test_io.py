import os

import nose.tools as nt
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import pandas
import fiona

from pygridtools import io
import testing


class test_loadBoundaryFromShapefile(object):
    def setup(self):
        self.shapefile = 'tests/test_data/simple_boundary.shp'
        self.known_df_columns = ['x', 'y', 'beta', 'upperleft',
        					     'reach', 'order']
        self.known_points_in_boundary = 19
        self.test_reach = 1
        self.known_points_in_testreach = 10

    def test_nofilter(self):
        df = io.loadBoundaryFromShapefile(self.shapefile)
        nt.assert_true(isinstance(df, pandas.DataFrame))
        nt.assert_list_equal(df.columns.tolist(), self.known_df_columns)
        nt.assert_equal(df.shape[0], self.known_points_in_boundary)

    def test_filter(self):
        df = io.loadBoundaryFromShapefile(
            self.shapefile,
            filterfxn=lambda r: r['properties']['reach'] == self.test_reach
        )
        nt.assert_equal(df.shape[0], self.known_points_in_testreach)


def test_dumpGridFile():
    grid = testing.makeSimpleGrid()
    outputfile = 'tests/result_files/grid.out'
    baselinefile = 'tests/baseline_files/grid.out'
    io.dumpGridFiles(grid, 'tests/result_files/grid.out')

    testing.compareTextFiles(outputfile, baselinefile)


class test_makeQuadCoords(object):
    def setup(self):
        x1 = 1
        x2 = 2
        y1 = 4
        y2 = 3
        z = 5

        self.xarr = np.array([[x1, x2], [x1, x2]])
        self.yarr = np.array([[y1, y1], [y2, y2]])
        self.zpnt = z
        self.mask = np.array([[False, False], [True, True]])

        self.known_base = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        self.known_no_masked = self.known_base.copy()
        self.known_masked = None
        self.known_with_z = np.array([
            [x1, y1, z], [x2, y1, z], [x2, y2, z], [x1, y2, z]
        ])

    def test_base(self):
        coords = io.makeQuadCoords(self.xarr, self.yarr)
        nptest.assert_array_equal(coords, self.known_base)

    def test_no_masked(self):
        xarr = np.ma.MaskedArray(self.xarr, mask=False)
        yarr = np.ma.MaskedArray(self.yarr, mask=False)
        coords = io.makeQuadCoords(xarr, yarr)
        nptest.assert_array_equal(coords, self.known_no_masked)

    def test_masked(self):
        xarr = np.ma.MaskedArray(self.xarr, mask=self.mask)
        yarr = np.ma.MaskedArray(self.yarr, mask=self.mask)
        coords = io.makeQuadCoords(xarr, yarr)
        nptest.assert_array_equal(coords, self.known_masked)

    def test_with_z(self):
        coords = io.makeQuadCoords(self.xarr, self.yarr, zpnt=self.zpnt)
        nptest.assert_array_equal(coords, self.known_with_z)


class test_makeRecord(object):
    def setup(self):
        self.point = [1, 2]
        self.point_array = np.array(self.point)
        self.non_point = [[1, 2], [5, 6], [5, 2]]
        self.non_point_array = np.array(self.non_point)
        self.mask = np.array([[1, 0], [0, 1], [1, 0]])
        self.masked_coords = np.ma.MaskedArray(self.non_point_array,
                                               mask=self.mask)

        self.props = {'prop1': 'this string', 'prop2': 3.1415}

        self.known_point = {
            'geometry': {
                'type': 'Point',
                'coordinates': [1, 2]
            },
            'id': 1,
            'properties': self.props
        }

        self.known_line = {
            'geometry': {
                'type': 'LineString',
                'coordinates': [[[1, 2], [5, 6], [5, 2]]]
            },
            'id': 1,
            'properties': self.props
        }

        self.known_polygon = {
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[1, 2], [5, 6], [5, 2]]]
            },
            'id': 1,
            'properties': self.props
        }

    def test_point(self):
        record = io.makeRecord(1, self.point, 'Point', self.props)
        nt.assert_dict_equal(record, self.known_point)

    def test_point_array(self):
        record = io.makeRecord(1, self.point_array, 'Point', self.props)
        nt.assert_dict_equal(record, self.known_point)

    def test_line(self):
        record = io.makeRecord(1, self.non_point, 'LineString', self.props)
        nt.assert_dict_equal(record, self.known_line)

    def test_line_array(self):
        record = io.makeRecord(1, self.non_point_array, 'LineString', self.props)
        nt.assert_dict_equal(record, self.known_line)

    def test_polygon(self):
        record = io.makeRecord(1, self.non_point, 'Polygon', self.props)
        nt.assert_dict_equal(record, self.known_polygon)

    def test_polygon_array(self):
        record = io.makeRecord(1, self.non_point_array, 'Polygon', self.props)
        nt.assert_dict_equal(record, self.known_polygon)

    @nt.raises(ValueError)
    def test_bad_geom(self):
        record = io.makeRecord(1, self.non_point_array, 'Circle', self.props)


class test_savePointShapefile(object):
    def setup(self):
        self.x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.y = np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])
        self.mask = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
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

    def test_with_arrays(self):
        fname = 'array_point.shp'
        outfile = os.path.join(self.outputdir, fname)
        basefile = os.path.join(self.baselinedir, fname)
        io.savePointShapefile(self.x, self.y, self.template, outfile,
                                   'w', river=self.river)

        testing.compareShapefiles(outfile, basefile)

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
        self.mask = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.template = 'tests/test_data/schema_template.shp'
        self.outputdir = 'tests/result_files'
        self.baselinedir = 'tests/baseline_files'
        self.river = 'test'
        self.maxDiff=None

    @nt.raises(ValueError)
    def test_bad_mode(self):
        io.saveGridShapefile(self.grid.x, self.grid.y,
                                  self.template, 'junk', 'r',
                                  elev=None)

    def test_with_arrays(self):
        fname = 'array_grid.shp'
        outfile = os.path.join(self.outputdir, fname)
        basefile = os.path.join(self.baselinedir, fname)
        io.saveGridShapefile(self.grid.x, self.grid.y, self.template,
                                  outfile, 'w', river=self.river,
                                  elev=None)

        testing.compareShapefiles(basefile, outfile, atol=0.001)


class test_shapefileToDataFrame(object):
    def setup(self):
        pass

    @nt.raises(NotImplementedError)
    def test_placeHolder(self):
        raise NotImplementedError


class test__write_cellinp(object):
    def setup(self):
        self.grid  = np.array([
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        ])
        self.basic_output = 'tests/result_files/cell_basic.inp'
        self.known_basic_output = 'tests/baseline_files/cell_basic.inp'
        self.chunked_output = 'tests/result_files/cell_chunked.inp'
        self.known_chunked_output = 'tests/baseline_files/cell_chunked.inp'
        self.triangle_output = 'tests/result_files/cell_triangle.inp'
        self.known_triangle_output = 'tests/baseline_files/cell_triangle.inp'

    @nt.raises(NotImplementedError)
    def test_basic(self):
        io._write_cellinp(self.grid, self.basic_output)
        testing.compareTextFiles(
            self.basic_output,
            self.known_basic_output
        )

    @nt.raises(NotImplementedError)
    def test_chunked(self):
        io._write_cellinp(self.grid, self.chunked_output, maxcols=5)
        testing.compareTextFiles(
            self.chunked_output,
            self.known_chunked_output
        )

    @nt.raises(NotImplementedError)
    def test_with_triangles(self):
        io._write_cellinp(self.grid, self.triangle_output, triangle_cells=True)
        testing.compareTextFiles(
            self.triangle_output,
            self.known_triangle_output
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
                                   'junk', river=self.river)
