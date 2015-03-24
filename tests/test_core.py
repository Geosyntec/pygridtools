import os


import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import pandas
import pygridgen

import nose.tools as nt
import numpy.testing as nptest
import pandas.util.testing as pdtest

import pygridgen
from pygridtools import core
import testing


class test__PointSet(object):
    def setup(self):
        from numpy import nan
        self.A = core._PointSet(np.arange(12).reshape(4, 3) * 1.0)
        self.B = core._PointSet(np.arange(8).reshape(2, 4) * 1.0)
        self.C = core._PointSet(np.arange(18).reshape(3, 6) * 1.0)

        self.known_flipped_A_points = np.array([
            [ 2.,  1.,  0.],
            [ 5.,  4.,  3.],
            [ 8.,  7.,  6.],
            [11., 10.,  9.]
        ])


        self.known_transposed_transformed_A_points = np.array([
            [2., 5., 8., 11.],
            [1., 4., 7., 10.],
            [0., 3., 6.,  9.],
        ])

        self.known_AB_vplus_0 = np.array([
            [ 0.,   1.,   2., nan],
            [ 3.,   4.,   5., nan],
            [ 6.,   7.,   8., nan],
            [ 9.,  10.,  11., nan],
            [ 0.,   1.,   2.,  3.],
            [ 4.,   5.,   6.,  7.]
        ])

        self.known_AB_vminus_0 = np.array([
            [ 0.,   1.,   2.,  3.],
            [ 4.,   5.,   6.,  7.],
            [ 0.,   1.,   2., nan],
            [ 3.,   4.,   5., nan],
            [ 6.,   7.,   8., nan],
            [ 9.,  10.,  11., nan]
        ])

        self.known_AB_vplus_2 = np.array([
            [ 0.,   1.,   2., nan,   nan, nan],
            [ 3.,   4.,   5., nan,   nan, nan],
            [ 6.,   7.,   8., nan,   nan, nan],
            [ 9.,  10.,  11., nan,   nan, nan],
            [nan,  nan,   0.,   1.,   2.,  3.],
            [nan,  nan,   4.,   5.,   6.,  7.]
        ])

        self.known_AB_vminus_2 = np.array([
            [nan, nan,  0.,  1.,  2.,  3.],
            [nan, nan,  4.,  5.,  6.,  7.],
            [ 0.,  1.,  2., nan, nan, nan],
            [ 3.,  4.,  5., nan, nan, nan],
            [ 6.,  7.,  8., nan, nan, nan],
            [ 9., 10., 11., nan, nan, nan]
        ])

        self.known_AB_vplus_neg1 = np.array([
            [nan, 0., 1.,   2.],
            [nan, 3., 4.,   5.],
            [nan, 6., 7.,   8.],
            [nan, 9., 10., 11.],
            [ 0., 1., 2.,   3.],
            [ 4., 5., 6.,   7.]
        ])

        self.known_AB_vminus_neg1 = np.array([
            [ 0., 1., 2.,   3.],
            [ 4., 5., 6.,   7.],
            [nan, 0., 1.,   2.],
            [nan, 3., 4.,   5.],
            [nan, 6., 7.,   8.],
            [nan, 9., 10., 11.]
        ])

        self.known_AB_hplus_0 = np.array([
            [0.,  1.,  2.,  0.,  1.,  2.,  3.],
            [3.,  4.,  5.,  4.,  5.,  6.,  7.],
            [6.,  7.,  8., nan, nan, nan, nan],
            [9., 10., 11., nan, nan, nan, nan]
        ])

        self.known_AB_hminus_0 = np.array([
            [ 0.,  1.,  2.,  3., 0.,  1.,  2.],
            [ 4.,  5.,  6.,  7., 3.,  4.,  5.],
            [nan, nan, nan, nan, 6.,  7.,  8.],
            [nan, nan, nan, nan, 9., 10., 11.]
        ])

        self.known_AB_hplus_2 = np.array([
            [0.,  1.,  2., nan, nan, nan, nan],
            [3.,  4.,  5., nan, nan, nan, nan],
            [6.,  7.,  8.,  0.,  1.,  2.,  3.],
            [9., 10., 11.,  4.,  5.,  6.,  7.]
        ])

        self.known_AB_hminus_2 = np.array([
            [nan, nan, nan, nan, 0.,  1.,  2.],
            [nan, nan, nan, nan, 3.,  4.,  5.],
            [ 0.,  1.,  2.,  3., 6.,  7.,  8.],
            [ 4.,  5.,  6.,  7., 9., 10., 11.]
        ])

        self.known_AB_hplus_neg1 = np.array([
            [nan, nan, nan,  0.,  1.,  2.,  3.],
            [ 0.,  1.,  2.,  4.,  5.,  6.,  7.],
            [ 3.,  4.,  5., nan, nan, nan, nan],
            [ 6.,  7.,  8., nan, nan, nan, nan],
            [ 9., 10., 11., nan, nan, nan, nan]
        ])

        self.known_AB_hminus_neg1 = np.array([
            [ 0.,  1.,  2.,  3., nan, nan, nan],
            [ 4.,  5.,  6.,  7.,  0.,  1.,  2.],
            [nan, nan, nan, nan,  3.,  4.,  5.],
            [nan, nan, nan, nan,  6.,  7.,  8.],
            [nan, nan, nan, nan,  9., 10., 11.]
        ])

    def test_points_and_setter(self):
        set_val = np.arange(16).reshape(4, 4) * 1
        self.A.points = set_val
        nptest.assert_array_equal(set_val, self.A.points)

    def test_transform(self):
        nptest.assert_array_equal(
            self.known_flipped_A_points,
            self.A.transform(np.fliplr).points
        )

    def test_transpose(self):
        nptest.assert_array_equal(
            self.A.points.T,
            self.A.transpose().points
        )

    def test_transpose_transform(self):
        nptest.assert_array_equal(
            self.known_transposed_transformed_A_points,
            self.A.transpose().transform(np.flipud).points
        )

    def test_transform_transpose(self):
        nptest.assert_array_equal(
            self.known_transposed_transformed_A_points,
            self.A.transpose().transform(np.flipud).points
        )

    def test_merge_vplus_0(self):
        nptest.assert_array_equal(
            self.known_AB_vplus_0,
            self.A.merge(self.B, how='v', where='+', shift=0).points,
        )

    def test_merge_vminus_0(self):
        nptest.assert_array_equal(
            self.known_AB_vminus_0,
            self.A.merge(self.B, how='v', where='-', shift=0).points,
        )

    def test_merge_vplus_2(self):
        nptest.assert_array_equal(
            self.known_AB_vplus_2,
            self.A.merge(self.B, how='v', where='+', shift=2).points,
        )

    def test_merge_vminus_2(self):
        nptest.assert_array_equal(
            self.known_AB_vminus_2,
            self.A.merge(self.B, how='v', where='-', shift=2).points,
        )

    def test_merge_vplus_neg1(self):
        nptest.assert_array_equal(
            self.known_AB_vplus_neg1,
            self.A.merge(self.B, how='v', where='+', shift=-1).points,
        )

    def test_merge_vminus_neg1(self):
        nptest.assert_array_equal(
            self.known_AB_vminus_neg1,
            self.A.merge(self.B, how='v', where='-', shift=-1).points,
        )

    def test_merge_hplus_0(self):
        nptest.assert_array_equal(
            self.known_AB_hplus_0,
            self.A.merge(self.B, how='h', where='+', shift=0).points,
        )

    def test_merge_hminus_0(self):
        nptest.assert_array_equal(
            self.known_AB_hminus_0,
            self.A.merge(self.B, how='h', where='-', shift=0).points,
        )

    def test_merge_hplus_2(self):
        nptest.assert_array_equal(
            self.known_AB_hplus_2,
            self.A.merge(self.B, how='h', where='+', shift=2).points,
        )

    def test_merge_hminus_2(self):
        nptest.assert_array_equal(
            self.known_AB_hminus_2,
            self.A.merge(self.B, how='h', where='-', shift=2).points,
        )

    def test_merge_hplus_neg1(self):
        nptest.assert_array_equal(
            self.known_AB_hplus_neg1,
            self.A.merge(self.B, how='h', where='+', shift=-1).points,
        )

    def test_merge_hminus_neg1(self):
        nptest.assert_array_equal(
            self.known_AB_hminus_neg1,
            self.A.merge(self.B, how='h', where='-', shift=-1).points,
        )


class test_ModelGrid(object):
    def setup(self):
        self.xn, self.yn = testing.makeSimpleNodes()
        self.xc, self.yc = testing.makeSimpleCells()
        self.g1 = core.ModelGrid(self.xn[:, :3], self.yn[:, :3])
        self.g2 = core.ModelGrid(self.xn[2:5, 3:], self.yn[2:5, 3:])

        self.known_rows = 9
        self.known_cols = 3
        self.known_df = pandas.DataFrame({
            ('easting', 0): {
                0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
                5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0
            }, ('easting', 1): {
                0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5,
                5: 1.5, 6: 1.5, 7: 1.5, 8: 1.5
            }, ('easting', 2): {
                0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0,
                5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0
            }, ('northing', 0): {
                0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0,
                5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0
            }, ('northing', 1): {
                0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0,
                5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0
            }, ('northing', 2): {
                0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0,
                5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0}
            })

        self.known_masked_cell_df = pandas.DataFrame({
            ('easting', 0): {
                0: nan, 1: nan, 2: 1.25, 3: 1.25, 4: 1.25,
                5: 1.25, 6: 1.25, 7: 1.25
            }, ('easting', 1): {
                0: nan, 1: nan, 2: 1.75, 3: 1.75, 4: 1.75,
                5: 1.75, 6: 1.75, 7: 1.75
            }, ('northing', 0): {
                0: nan, 1: nan, 2: 1.25, 3: 1.75, 4: 2.25,
                5: 2.75, 6: 3.25, 7: 3.75
            }, ('northing', 1): {
                0: nan, 1: nan, 2: 1.25, 3: 1.75, 4: 2.25,
                5: 2.75, 6: 3.25, 7: 3.75
            }
        })


        self.known_coord_pairs = np.array([
            [1. , 0. ], [1.5, 0. ], [2. , 0. ], [1. , 0.5],
            [1.5, 0.5], [2. , 0.5], [1. , 1. ], [1.5, 1. ],
            [2. , 1. ], [1. , 1.5], [1.5, 1.5], [2. , 1.5],
            [1. , 2. ], [1.5, 2. ], [2. , 2. ], [1. , 2.5],
            [1.5, 2.5], [2. , 2.5], [1. , 3. ], [1.5, 3. ],
            [2. , 3. ], [1. , 3.5], [1.5, 3.5], [2. , 3.5],
            [1. , 4. ], [1.5, 4. ], [2. , 4. ]
        ])

        self.known_mask = np.array([
            [True,  True ], [True,  True ],
            [False, False], [False, False],
            [False, False], [False, False],
            [False, False], [False, False],
        ])

        self.known_node_pairs_masked = np.array([
            [ nan,  nan], [ nan,  nan], [ nan,  nan], [ nan,  nan],
            [1.25, 1.25], [1.75, 1.25], [1.25, 1.75], [1.75, 1.75],
            [1.25, 2.25], [1.75, 2.25], [1.25, 2.75], [1.75, 2.75],
            [1.25, 3.25], [1.75, 3.25], [1.25, 3.75], [1.75, 3.75]
        ])

        self.known_node_pairs = np.array([
            [1.25, 0.25], [1.75, 0.25], [1.25, 0.75], [1.75, 0.75],
            [1.25, 1.25], [1.75, 1.25], [1.25, 1.75], [1.75, 1.75],
            [1.25, 2.25], [1.75, 2.25], [1.25, 2.75], [1.75, 2.75],
            [1.25, 3.25], [1.75, 3.25], [1.25, 3.75], [1.75, 3.75]
        ])

    @nt.raises(ValueError)
    def test_bad_input(self):
        mg = core.ModelGrid(self.xc, self.yc[2:, 2:])

    def test_nodes_x(self):
        nt.assert_true(hasattr(self.g1, 'nodes_x'))
        nt.assert_true(isinstance(self.g1.nodes_x, core._PointSet))

    def test_nodes_y(self):
        nt.assert_true(hasattr(self.g1, 'nodes_y'))
        nt.assert_true(isinstance(self.g1.nodes_y, core._PointSet))

    def test_cells_x(self):
        nt.assert_true(hasattr(self.g1, 'cells_x'))
        nt.assert_true(isinstance(self.g1.cells_x, np.ndarray))
        nptest.assert_array_equal(self.g1.cells_x, self.xc[:, :2])

    def test_cells_y(self):
        nt.assert_true(hasattr(self.g1, 'cells_y'))
        nt.assert_true(isinstance(self.g1.cells_y, np.ndarray))
        nptest.assert_array_equal(self.g1.cells_y, self.yc[:, :2])

    def test_icells(self):
        nt.assert_equal(
            self.g1.icells,
            self.known_cols - 1
        )

    def test_jcells(self):
        nt.assert_equal(
            self.g1.jcells,
            self.known_rows - 1
        )

    def test_inodes(self):
        nt.assert_equal(
            self.g1.inodes,
            self.known_cols
        )

    def test_jnodes(self):
        nt.assert_equal(
            self.g1.jnodes,
            self.known_rows
        )

    def test_shape(self):
        nt.assert_true(hasattr(self.g1, 'shape'))
        nt.assert_tuple_equal(
            self.g1.shape,
            (self.known_rows, self.known_cols)
        )

    def test_cell_shape(self):
        nt.assert_true(hasattr(self.g1, 'cell_shape'))
        nt.assert_tuple_equal(
            self.g1.cell_shape,
            (self.known_rows - 1, self.known_cols - 1)
        )

    def test_cell_mask(self):
        nt.assert_true(hasattr(self.g1, 'cell_mask'))
        known_base_mask = np.array([
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
        ])
        nptest.assert_array_equal(self.g1.cell_mask, known_base_mask)

    def test_template(self):
        nt.assert_equal(self.g1.template, None)

        template_value = 'junk'
        self.g1.template = template_value
        nt.assert_equal(self.g1.template, template_value)

    def test_as_dataframe_nomask_nodes(self):
        pdtest.assert_frame_equal(
            self.g1.as_dataframe(usemask=False, which='nodes'),
            self.known_df,
            check_names=False
        )

        pdtest.assert_index_equal(
            self.g1.as_dataframe().columns,
            self.known_df.columns,
        )

    def test_as_coord_pairs_nomask_nodes(self):
        nptest.assert_array_equal(
            self.g1.as_coord_pairs(usemask=False, which='nodes'),
            self.known_coord_pairs
        )

    @nt.raises(ValueError)
    def test_as_dataframe_mask_nodes(self):
        self.g1.as_dataframe(usemask=True, which='nodes')

    @nt.raises(ValueError)
    def test_as_coord_pairs_mask_nodes(self):
        self.g1.as_coord_pairs(usemask=True, which='nodes')

    def test_as_coord_pairs_nomask_cells(self):
        nptest.assert_array_equal(
            self.g1.as_coord_pairs(usemask=False, which='cells'),
            self.known_node_pairs
        )

    def test_as_coord_pairs_mask_cells(self):
        self.g1.cell_mask = self.known_mask
        nptest.assert_array_equal(
            self.g1.as_coord_pairs(usemask=True, which='cells'),
            self.known_node_pairs_masked
        )

    def test_as_dataframe_mask_cells(self):
        self.g1.cell_mask = self.known_mask
        df = self.g1.as_dataframe(usemask=True, which='cells')
        pdtest.assert_frame_equal(df, self.known_masked_cell_df,
                                  check_names=False)

        pdtest.assert_index_equal(
            df.columns, self.known_masked_cell_df.columns,
        )

    def test_transform(self):
        gx = self.g1.xn.copy() * 10
        g = self.g1.transform(lambda x: x * 10)
        nptest.assert_array_equal(g.xn, gx)

    def test_transpose(self):
        gx = self.g1.xn.copy()
        nptest.assert_array_equal(
            self.g1.transpose().xn,
            gx.transpose()
        )

    def test_fliplr(self):
        gx = np.fliplr(self.g1.xn.copy())
        g = self.g1.fliplr()
        nptest.assert_array_equal(g.xn, gx)

    def test_flipud(self):
        gx = np.flipud(self.g1.xn.copy())
        g = self.g1.flipud()
        nptest.assert_array_equal(g.xn, gx)

    def test_merge(self):
        g3 = self.g1.merge(self.g2, how='horiz', where='+', shift=2)
        g4 = core.ModelGrid(self.xn, self.yn)

        nptest.assert_array_equal(g3.xn, g4.xn)
        nptest.assert_array_equal(g3.xc, g4.xc)


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
        grid, junk = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            **self.gridparams
        )
        nt.assert_equal(junk, None)
        nt.assert_true(isinstance(grid, pygridgen.Gridgen))

    def test_with_plot_without_fig_path(self):
        grid, fig = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=True,
            makegrid=True,
            **self.gridparams
        )
        nt.assert_true(isinstance(fig, plt.Figure))
        fig.savefig('tests/result_images/grid_basic.png', dpi=150)

    def test_with_plot_with_xlimits_autosaved(self):
        grid, fig = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=True,
            makegrid=True,
            xlimits=[0, 20],
            figpath='tests/result_images/grid_autosaved_xlims.png',
            **self.gridparams
        )
        nt.assert_true(isinstance(fig, plt.Figure))

    def test_with_plot_with_ax(self):
        fig, ax = plt.subplots()
        grid, fig1 = core.makeGrid(
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
        grid1, junk = core.makeGrid(
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
        grid, junk = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            **self.gridparams
        )

    @nt.raises(ValueError)
    def test_makegrid_no_ny(self):
        ny = self.gridparams.pop('ny')
        grid, junk = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            **self.gridparams
        )

    @nt.raises(ValueError)
    def test_False_makegrid_None_grid(self):
        grid, junk = core.makeGrid(
            coords=None,
            bathydata=self.bathy,
            plot=False,
            makegrid=False,
            grid=None,
            **self.gridparams
        )

    def test_check_gridparams(self):
        grid, junk = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            **self.gridparams
        )
        for gpkey in self.gridparams:
            nt.assert_equal(self.gridparams[gpkey], getattr(grid, gpkey))

    def test_check_GEFDC_output_bathy(self):
        grid, junk = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            outdir='tests/result_files/extra',
            title='Extra Test Title',
            **self.gridparams
        )

        bathyfile = 'depdat.inp'
        gefdcfile = 'gefdc.inp'

        outputdir = 'tests/result_files/extra'
        baselinedir = 'tests/baseline_files/extra'

        testing.compareTextFiles(
            os.path.join(outputdir, bathyfile),
            os.path.join(baselinedir, bathyfile)
        )

    def test_check_GEFDC_output_controlfile(self):
        grid, junk = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            plot=False,
            makegrid=True,
            outdir='tests/result_files/extra',
            title='Extra Test Title',
            **self.gridparams
        )

        bathyfile = 'depdat.inp'
        gefdcfile = 'gefdc.inp'

        outputdir = 'tests/result_files/extra'
        baselinedir = 'tests/baseline_files/extra'
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


