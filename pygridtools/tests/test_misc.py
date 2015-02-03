import os

import nose.tools as nt
import numpy as np
import numpy.testing as nptest
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas
import pygridgen

from pygridtools import misc
import testing


class test_interpolateBathymetry(object):
    def setup(self):
        self.bathy = testing.makeSimpleBathy()
        self.grid = testing.makeSimpleGrid()

        self.known_real_elev = np.ma.MaskedArray(
            data= [
                [100.15, 100.19, -999.99, -999.99, -999.99, -999.99],
                [100.20, 100.25,  100.64,  100.73,  100.85,  100.94],
                [100.25, 100.30,  100.35,  100.39,  100.44,  100.50],
                [100.30, 100.35,  100.39,  100.44,  100.49,  100.55],
                [100.34, 100.40, -999.99, -999.99, -999.99, -999.99],
                [100.39, 100.45, -999.99, -999.99, -999.99, -999.99],
                [100.44, 100.49, -999.99, -999.99, -999.99, -999.99],
                [100.50, 100.54, -999.99, -999.99, -999.99, -999.99]
            ],
            mask=[
                [False, False,  True,  True,  True,  True],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False,  True,  True,  True,  True],
                [False, False,  True,  True,  True,  True],
                [False, False,  True,  True,  True,  True],
                [False, False,  True,  True,  True,  True]
            ]
        )

    def test_fake_bathy(self):
        misc.interpolateBathymetry(None, self.grid)
        nptest.assert_array_equal(
            self.grid.elev,
            np.ma.MaskedArray(data=np.zeros(self.grid.x_rho.shape),
                              mask=self.grid.x_rho.mask)
        )
        nt.assert_tuple_equal(self.grid.elev.shape, self.grid.x_rho.shape)

    def test_real_bathy(self):
        misc.interpolateBathymetry(self.bathy, self.grid)
        nptest.assert_array_almost_equal(
            self.grid.elev, self.known_real_elev, decimal=2
        )

    def teardown(self):
        pass


class test_makeGrid(object):
    def setup(self):
        self.coords = testing.makeSimpleBoundary()
        self.bathy = testing.makeSimpleBathy()
        self.gridparams = {
            'nx': 9,
            'ny': 7,
            'nnodes': 12,
            'verbose': False,
            'ul_idx': 0
        }

    def test_with_coords_and_bathy(self):
        grid, junk = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            **self.gridparams
        )
        nt.assert_equal(junk, None)
        nt.assert_true(isinstance(grid, pygridgen.Gridgen))

    def test_with_plot_without_fig_path(self):
        grid, fig = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=True,
            makegrid=True,
            **self.gridparams
        )
        nt.assert_true(isinstance(fig, plt.Figure))
        fig.savefig('pygridtools/tests/result_images/grid_basic.png', dpi=150)

    def test_with_plot_with_xlimits_autosaved(self):
        grid, fig = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=True,
            makegrid=True,
            xlimits=[0, 20],
            figpath='pygridtools/tests/result_images/grid_autosaved_xlims.png',
            **self.gridparams
        )
        nt.assert_true(isinstance(fig, plt.Figure))

    def test_with_plot_with_ax(self):
        fig, ax = plt.subplots()
        grid, fig1 = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=True, ax=ax,
            makegrid=True,
            **self.gridparams
        )
        nt.assert_true(isinstance(fig, plt.Figure))
        nt.assert_equal(fig, fig1)

    def test_with_grid(self):
        grid = testing.makeSimpleGrid()
        grid1, junk = misc.makeGrid(
            plot=False,
            makegrid=False,
            grid=grid,
            **self.gridparams
        )
        nt.assert_equal(junk, None)
        nt.assert_equal(grid, grid1)
        nt.assert_true(isinstance(grid, pygridgen.Gridgen))
        nt.assert_true(isinstance(grid1, pygridgen.Gridgen))
        pass


    @nt.raises(ValueError)
    def test_makegrid_no_nx(self):
        nx = self.gridparams.pop('nx')
        grid, junk = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            **self.gridparams
        )

    @nt.raises(ValueError)
    def test_makegrid_no_ny(self):
        ny = self.gridparams.pop('ny')
        grid, junk = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            **self.gridparams
        )

    @nt.raises(ValueError)
    def test_False_makegrid_None_grid(self):
        grid, junk = misc.makeGrid(
            coords=None,
            bathydata=self.bathy,
            plot=False,
            makegrid=False,
            grid=None,
            **self.gridparams
        )

    def test_check_gridparams(self):
        grid, junk = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            **self.gridparams
        )
        for gpkey in self.gridparams:
            nt.assert_equal(self.gridparams[gpkey], getattr(grid, gpkey))

    def test_check_GEFDC_output_bathy(self):
        grid, junk = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            outdir='pygridtools/tests/result_files/extra',
            title='Extra Test Title',
            **self.gridparams
        )

        bathyfile = 'depdat.inp'
        gefdcfile = 'gefdc.inp'

        outputdir = 'pygridtools/tests/result_files/extra'
        baselinedir = 'pygridtools/tests/baseline_files/extra'

        testing.compareTextFiles(
            os.path.join(outputdir, bathyfile),
            os.path.join(baselinedir, bathyfile)
        )

    def test_check_GEFDC_output_controlfile(self):
        grid, junk = misc.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            outdir='pygridtools/tests/result_files/extra',
            title='Extra Test Title',
            **self.gridparams
        )

        bathyfile = 'depdat.inp'
        gefdcfile = 'gefdc.inp'

        outputdir = 'pygridtools/tests/result_files/extra'
        baselinedir = 'pygridtools/tests/baseline_files/extra'
        testing.compareTextFiles(
            os.path.join(outputdir, gefdcfile),
            os.path.join(baselinedir, gefdcfile)
        )

    def teardown(self):
        plt.close('all')
        try:
            os.remove('gefdc.inp')
            os.remove('depdat.inp')
        except:
            pass


class test__proccess_array(object):
    def setup(self):
        np.random.seed(0)
        self.tall = np.random.normal(size=(37, 4))
        self.wide = self.tall.T
        self.mask = np.random.randint(0, high=2, size=(37, 4))

        self.tall_masked = np.ma.MaskedArray(data=self.tall, mask=self.mask)
        self.wide_masked = np.ma.MaskedArray(data=self.wide, mask=self.mask)
        pass

    def test_no_op(self):
        arr2 = misc._process_array(self.tall, False)
        nptest.assert_array_equal(arr2, self.tall)

    def test_no_op_masked(self):
        arr2 = misc._process_array(self.tall_masked, False)
        nptest.assert_array_equal(arr2, self.tall)

    def test_transpose(self):
        arr2 = misc._process_array(self.wide, True)
        nptest.assert_array_equal(arr2, self.tall)

    def test_transpose_mask(self):
        arr2 = misc._process_array(self.wide_masked, True)
        nptest.assert_array_equal(arr2, self.tall)

    def test_with_transform_flip(self):
        trans = np.flipud
        arr2 = misc._process_array(self.wide, True, transform=trans)
        nptest.assert_array_equal(arr2, trans(self.tall))

    def test_transform_numeric(self):
        trans = lambda x: x + 2
        arr2 = misc._process_array(self.wide, True, transform=trans)
        nptest.assert_array_equal(arr2, trans(self.tall))

    def teardown(self):
        pass


class test__grid_attr_to_df(object):
    def setup(self):
        np.random.seed(0)
        self.ny = 37
        self.nx = 8

        size = (self.ny, self.nx)
        self.xattr = np.random.random_integers(low=0, high=100, size=size)
        self.yattr = np.random.random_integers(low=0, high=100, size=size)
        self.junk = np.random.random_integers(low=0, high=100, size=(27,10))

    @nt.raises(ValueError)
    def test_bad_shapes(self):
        misc._grid_attr_to_df(self.xattr, self.junk, False)

    def test_basic(self):
        df = misc._grid_attr_to_df(self.xattr, self.yattr, False)
        nt.assert_tuple_equal(df.shape, (self.ny, self.nx*2))

    def test_stock_i_index(self):
        df = misc._grid_attr_to_df(self.xattr, self.yattr, False)
        expected_i = list(range(self.nx))
        returned_i = pandas.unique(df.columns.get_level_values(1)).tolist()
        nt.assert_list_equal(expected_i, returned_i)

    def test_stock_j_index(self):
        df = misc._grid_attr_to_df(self.xattr, self.yattr, False)
        expected_j = list(range(self.ny))
        returned_j = pandas.unique(df.index.get_level_values('j')).tolist()
        nt.assert_list_equal(expected_j, returned_j)

    def test_offset_transposed_i_index(self):
        df = misc._grid_attr_to_df(self.xattr, self.yattr, True)
        expected_i = list(range(self.ny))
        returned_i = pandas.unique(df.columns.get_level_values(1)).tolist()
        nt.assert_list_equal(expected_i, returned_i)

    def test_offset_transposed_j_index(self):
        df = misc._grid_attr_to_df(self.xattr, self.yattr, True)
        expected_j = list(range(self.nx))
        returned_j = pandas.unique(df.index.get_level_values('j')).tolist()
        nt.assert_list_equal(expected_j, returned_j)

    def teardown(self):
        pass


class test_GridDataFrame(object):
    def setup(self):
        self.grid = testing.makeSimpleGrid()
        self.gdf = misc.GridDataFrame(self.grid)
        self.other_gdf = misc.GridDataFrame(self.grid)
        self.top_col_level = ['northing', 'easting']
        self.df_attrs = ['u', 'v', 'centers', 'psi', 'verts']
        self.template = 'pygridtools/tests/test_data/schema_template.shp'
        self.point_baseline = 'pygridtools/tests/baseline_files/gdf_point.shp'
        self.point_output = 'pygridtools/tests/result_files/gdf_point.shp'
        self.polygon_baseline = 'pygridtools/tests/baseline_files/gdf_polygon.shp'
        self.polygon_output = 'pygridtools/tests/result_files/gdf_polygon.shp'

    def test_dfs(self):
        for a in self.df_attrs:
            nt.assert_true(isinstance(getattr(self.gdf, a), pandas.DataFrame))

    def test_other_attr(self):
        nt.assert_equal(False, self.gdf.transpose)
        nt.assert_equal(self.grid, self.gdf.grid)
        nt.assert_list_equal(self.gdf.merged_grids, [])

    def test_col_levels(self):
        for a in self.df_attrs:
            for col in self.top_col_level:
                nt.assert_true(col in getattr(self.gdf, a).columns.get_level_values(0))

    def test_merge_grid_list(self):
        self.gdf.mergeGrid(self.other_gdf, 'j', '+', self.grid.ny)
        nt.assert_list_equal(self.gdf.merged_grids, [self.other_gdf])

    def test_writeGridOut(self):
        outputfile = 'pygridtools/tests/result_files/gdf2grid.out'
        baselinefile = 'pygridtools/tests/baseline_files/grid.out'

        self.gdf.writeGridOut(outputfile)

        testing.compareTextFiles(outputfile, baselinefile)

    def test_writeVertsToShapefile_Point(self):
        self.gdf.writeVertsToShapefile(
            self.point_output,
            geomtype='Point',
            template=self.template
        )

        testing.compareShapefiles(
            self.point_output, self.point_baseline, atol=0.1
        )

    def test_writeVertsToShapefile_Polygon(self):
        self.gdf.writeVertsToShapefile(
            self.polygon_output,
            geomtype='Polygon',
            template=self.template
        )

        testing.compareShapefiles(
            self.polygon_output, self.polygon_baseline, atol=0.1
        )

    def teardown(self):
        pass


class test_mergePoints(object):
    def setup(self):

        self.df0 = pandas.DataFrame(np.array([
            [13.7, 13.8],
            [14.7, 14.8],
            [15.7, 15.8],
            [16.7, 16.8],
            [17.7, 17.8],
        ]))

        self.df1 = pandas.DataFrame(np.array([
            [ 6.6,  6.7,  6.8],
            [ 7.6,  7.7,  7.8],
            [ 8.6,  8.7,  8.8],
            [ 9.6,  9.7,  9.8],
            [10.6, 10.7, 10.8],
            [11.6, 11.7, 11.8],
            [12.6, 12.7, 12.8],
        ]))

        self.df2 = pandas.DataFrame(np.array([
            [7.9, 7.10, 7.11, 7.12, 7.13],
            [8.9, 8.10, 8.11, 8.12, 8.13],
            [9.9, 9.10, 9.11, 9.12, 9.13],
        ]))

        self.df3 = pandas.DataFrame(np.array([
            [1.4, 1.5, 1.6, 1.7, 1.8],
            [2.4, 2.5, 2.6, 2.7, 2.8],
            [3.4, 3.5, 3.6, 3.7, 3.8],
            [4.4, 4.5, 4.6, 4.7, 4.8],
            [5.4, 5.5, 5.6, 5.7, 5.8],
        ]))

        self.df4 = pandas.DataFrame(np.array([
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
        ]))

        self.df5 = pandas.DataFrame(np.array([
            [7.14, 7.15, 7.16],
            [8.14, 8.15, 8.16],
        ]))

        self.expected_df1and2 = pandas.DataFrame(np.array([
            [ 6.6,  6.7,  6.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [ 7.6,  7.7,  7.8,    7.9,   7.10,   7.11,   7.12,   7.13],
            [ 8.6,  8.7,  8.8,    8.9,   8.10,   8.11,   8.12,   8.13],
            [ 9.6,  9.7,  9.8,    9.9,   9.10,   9.11,   9.12,   9.13],
            [10.6, 10.7, 10.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [11.6, 11.7, 11.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [12.6, 12.7, 12.8, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]))

        self.expected_df0and1 = pandas.DataFrame(np.array([
            [   6.6,  6.7,  6.8],
            [   7.6,  7.7,  7.8],
            [   8.6,  8.7,  8.8],
            [   9.6,  9.7,  9.8],
            [  10.6, 10.7, 10.8],
            [  11.6, 11.7, 11.8],
            [  12.6, 12.7, 12.8],
            [np.nan, 13.7, 13.8],
            [np.nan, 14.7, 14.8],
            [np.nan, 15.7, 15.8],
            [np.nan, 16.7, 16.8],
            [np.nan, 17.7, 17.8],
        ]))

        self.expected_df0and1and2 = pandas.DataFrame(np.array([
            [   6.6,  6.7,  6.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [   7.6,  7.7,  7.8,    7.9,   7.10,   7.11,   7.12,   7.13],
            [   8.6,  8.7,  8.8,    8.9,   8.10,   8.11,   8.12,   8.13],
            [   9.6,  9.7,  9.8,    9.9,   9.10,   9.11,   9.12,   9.13],
            [  10.6, 10.7, 10.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [  11.6, 11.7, 11.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [  12.6, 12.7, 12.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 13.7, 13.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 14.7, 14.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 15.7, 15.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 16.7, 16.8, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 17.7, 17.8, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]))

        self.expected_df1and3 = pandas.DataFrame(np.array([
            [   1.4,     1.5,  1.6,  1.7,  1.8],
            [   2.4,     2.5,  2.6,  2.7,  2.8],
            [   3.4,     3.5,  3.6,  3.7,  3.8],
            [   4.4,     4.5,  4.6,  4.7,  4.8],
            [   5.4,     5.5,  5.6,  5.7,  5.8],
            [np.nan,  np.nan,  6.6,  6.7,  6.8],
            [np.nan,  np.nan,  7.6,  7.7,  7.8],
            [np.nan,  np.nan,  8.6,  8.7,  8.8],
            [np.nan,  np.nan,  9.6,  9.7,  9.8],
            [np.nan,  np.nan, 10.6, 10.7, 10.8],
            [np.nan,  np.nan, 11.6, 11.7, 11.8],
            [np.nan,  np.nan, 12.6, 12.7, 12.8],
        ]))

        self.expected_df3and4 = pandas.DataFrame(np.array([
            [   0.0,    0.1,    0.2,    0.3, np.nan, np.nan, np.nan, np.nan, np.nan],
            [   1.0,    1.1,    1.2,    1.3,    1.4,    1.5,    1.6,    1.7,    1.8],
            [   2.0,    2.1,    2.2,    2.3,    2.4,    2.5,    2.6,    2.7,    2.8],
            [   3.0,    3.1,    3.2,    3.3,    3.4,    3.5,    3.6,    3.7,    3.8],
            [np.nan, np.nan, np.nan, np.nan,    4.4,    4.5,    4.6,    4.7,    4.8],
            [np.nan, np.nan, np.nan, np.nan,    5.4,    5.5,    5.6,    5.7,    5.8],
        ]))

        self.expected_all_dfs = pandas.DataFrame(np.array([
            [   0.0,    0.1,    0.2,    0.3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [   1.0,    1.1,    1.2,    1.3,    1.4,    1.5,    1.6,    1.7,    1.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [   2.0,    2.1,    2.2,    2.3,    2.4,    2.5,    2.6,    2.7,    2.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [   3.0,    3.1,    3.2,    3.3,    3.4,    3.5,    3.6,    3.7,    3.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan,    4.4,    4.5,    4.6,    4.7,    4.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan,    5.4,    5.5,    5.6,    5.7,    5.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,    6.6,    6.7,    6.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,    7.6,    7.7,    7.8,    7.9,   7.10,   7.11,   7.12,   7.13,   7.14,   7.15,   7.16],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,    8.6,    8.7,    8.8,    8.9,   8.10,   8.11,   8.12,   8.13,   8.14,   8.15,   8.16],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,    9.6,    9.7,    9.8,    9.9,   9.10,   9.11,   9.12,   9.13, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,   10.6,   10.7,   10.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,   11.6,   11.7,   11.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,   12.6,   12.7,   12.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,   13.7,   13.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,   14.7,   14.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,   15.7,   15.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,   16.7,   16.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,   17.7,   17.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]))

    def test_vertical_merge_above(self):
        df_merged = misc.mergePoints(self.df1, self.df3, how='j', where='-', offset=-2)
        nptest.assert_array_equal(df_merged.values, self.expected_df1and3.values)

    def test_vertical_merge_below(self):
        df_merged = misc.mergePoints(self.df3, self.df1, how='j', where='+', offset=2)
        nptest.assert_array_equal(df_merged.values, self.expected_df1and3.values)

    def test_horizontal_merge_left(self):
        df_merged = misc.mergePoints(self.df2, self.df1, how='i', where='-', offset=-1)
        nptest.assert_array_equal(df_merged.values, self.expected_df1and2.values)

    def test_horizontal_merge_right(self):
        df_merged = misc.mergePoints(self.df1, self.df2, how='i', where='+', offset=1)
        nptest.assert_array_equal(df_merged.values, self.expected_df1and2.values)

    def test_vert_merge_0and1(self):
        merged = misc.mergePoints(self.df0, self.df1, how='j', where='-', offset=-1)
        nptest.assert_array_equal(merged.values, self.expected_df0and1.values)

    def test_vert_merge_0and1and2(self):
        step1 = misc.mergePoints(self.df0, self.df1, how='j', where='-', offset=-1)
        step2 = misc.mergePoints(step1, self.df2, how='i', where='+', offset=1)
        nptest.assert_array_equal(step2.values, self.expected_df0and1and2.values)

    def test_big_grid_lowerleft_to_upperright(self):
        step1 = misc.mergePoints(self.df4, self.df3, how='i', where='+', offset=1)
        step2 = misc.mergePoints(step1, self.df1, how='j', where='+', offset=6)
        step3 = misc.mergePoints(step2, self.df2, how='i', where='+', offset=7)
        step4 = misc.mergePoints(step3, self.df5, how='i', where='+', offset=7)
        step5 = misc.mergePoints(step4, self.df0, how='j', where='+', offset=7)

        nptest.assert_array_equal(step5.values, self.expected_all_dfs.values)

    def test_big_grid_upperright_to_lowerleft(self):
        step1 = misc.mergePoints(self.df0, self.df1, how='j', where='-', offset=-1)
        step2 = misc.mergePoints(step1, self.df2, how='i', where='+', offset=1)
        step3 = misc.mergePoints(step2, self.df3, how='j', where='-', offset=-2)
        step4 = misc.mergePoints(step3, self.df4, how='i', where='-', offset=-1)
        step5 = misc.mergePoints(step4, self.df5, how='i', where='+', offset=7)

        nptest.assert_array_equal(step5.values, self.expected_all_dfs.values)


class test__add_second_col_level(object):
    def setup(self):
        self.index = list('abcd')
        self.columns = list('xyz')
        self.new_level = 'test'
        self.df = pandas.DataFrame(index=self.index, columns=self.columns)
        self.known_columns = pandas.MultiIndex.from_product([[self.new_level], self.columns])

    def test_basic(self):
        newdf = misc._add_second_col_level(self.new_level, self.df)
        nptest.assert_array_equal(newdf.columns, self.known_columns)

    @nt.raises(ValueError)
    def test_existing_multi_col(self):
        newdf = misc._add_second_col_level(self.new_level, self.df)
        newdf2 = misc._add_second_col_level(self.new_level, newdf)
