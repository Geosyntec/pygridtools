import os


import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas
import pygridgen

import nose.tools as nt
import numpy.testing as nptest
import pandas.util.testing as pdtest

from pygridtools import misc
import testing

np.set_printoptions(linewidth=150, nanstr='-')


class test_interpolateBathymetry(object):
    def setup(self):
        self.bathy = testing.makeSimpleBathy()
        self.grid = testing.makeSimpleGrid()

        self.known_real_elev = np.ma.MaskedArray(
            data= [
                [100.15, 100.2 , -999.99, -999.99, -999.99, -999.99],
                [100.2 , 100.25,  100.65,  100.74,  100.83,  100.95],
                [100.25, 100.3 ,  100.35,  100.4 ,  100.45,  100.5 ],
                [100.3 , 100.35,  100.4 ,  100.45,  100.5 ,  100.55],
                [100.35, 100.4 , -999.99, -999.99, -999.99, -999.99],
                [100.4 , 100.45, -999.99, -999.99, -999.99, -999.99],
                [100.45, 100.5 , -999.99, -999.99, -999.99, -999.99],
                [100.5 , 100.55, -999.99, -999.99, -999.99, -999.99]
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
        fig.savefig('tests/result_images/grid_basic.png', dpi=150)

    def test_with_plot_with_xlimits_autosaved(self):
        grid, fig = misc.makeGrid(
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
        grid, junk = misc.makeGrid(
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


class test_padded_stack(object):

    def setup(self):
        from numpy import nan
        self.g0 = np.array([
            [13.7, 13.8],
            [14.7, 14.8],
            [15.7, 15.8],
            [16.7, 16.8],
            [17.7, 17.8],
        ])

        self.g1 = np.array([
            [ 6.6,  6.7,  6.8],
            [ 7.6,  7.7,  7.8],
            [ 8.6,  8.7,  8.8],
            [ 9.6,  9.7,  9.8],
            [10.6, 10.7, 10.8],
            [11.6, 11.7, 11.8],
            [12.6, 12.7, 12.8],
        ])

        self.g2 = np.array([
            [7.9, 7.10, 7.11, 7.12, 7.13],
            [8.9, 8.10, 8.11, 8.12, 8.13],
            [9.9, 9.10, 9.11, 9.12, 9.13],
        ])

        self.g3 = np.array([
            [1.4, 1.5, 1.6, 1.7, 1.8],
            [2.4, 2.5, 2.6, 2.7, 2.8],
            [3.4, 3.5, 3.6, 3.7, 3.8],
            [4.4, 4.5, 4.6, 4.7, 4.8],
            [5.4, 5.5, 5.6, 5.7, 5.8],
        ])

        self.g4 = np.array([
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
        ])

        self.g5 = np.array([
            [7.14, 7.15, 7.16],
            [8.14, 8.15, 8.16],
        ])

        self.expected_g1_2_left = np.array([
            [nan,  nan,  nan,  nan,  nan,  6.6,  6.7,  6.8],
            [7.9, 7.10, 7.11, 7.12, 7.13,  7.6,  7.7,  7.8],
            [8.9, 8.10, 8.11, 8.12, 8.13,  8.6,  8.7,  8.8],
            [9.9, 9.10, 9.11, 9.12, 9.13,  9.6,  9.7,  9.8],
            [nan,  nan,  nan,  nan,  nan, 10.6, 10.7, 10.8],
            [nan,  nan,  nan,  nan,  nan, 11.6, 11.7, 11.8],
            [nan,  nan,  nan,  nan,  nan, 12.6, 12.7, 12.8],
        ])

        self.expected_g1_2_right = np.array([
            [ 6.6,  6.7,  6.8, nan,  nan,  nan,  nan,  nan],
            [ 7.6,  7.7,  7.8, 7.9, 7.10, 7.11, 7.12, 7.13],
            [ 8.6,  8.7,  8.8, 8.9, 8.10, 8.11, 8.12, 8.13],
            [ 9.6,  9.7,  9.8, 9.9, 9.10, 9.11, 9.12, 9.13],
            [10.6, 10.7, 10.8, nan,  nan,  nan,  nan,  nan],
            [11.6, 11.7, 11.8, nan,  nan,  nan,  nan,  nan],
            [12.6, 12.7, 12.8, nan,  nan,  nan,  nan,  nan],
        ])

        self.expected_g0_1 = np.array([
            [ nan,  6.6,  6.7,  6.8],
            [ nan,  7.6,  7.7,  7.8],
            [ nan,  8.6,  8.7,  8.8],
            [ nan,  9.6,  9.7,  9.8],
            [ nan, 10.6, 10.7, 10.8],
            [ nan, 11.6, 11.7, 11.8],
            [ nan, 12.6, 12.7, 12.8],
            [13.7, 13.8, nan,  nan],
            [14.7, 14.8, nan,  nan],
            [15.7, 15.8, nan,  nan],
            [16.7, 16.8, nan,  nan],
            [17.7, 17.8, nan,  nan],
        ])

        self.expected_g0_1_2 = np.array([
            [ 6.6,  6.7,  6.8, nan,  nan,  nan,  nan,  nan],
            [ 7.6,  7.7,  7.8, 7.9, 7.10, 7.11, 7.12, 7.13],
            [ 8.6,  8.7,  8.8, 8.9, 8.10, 8.11, 8.12, 8.13],
            [ 9.6,  9.7,  9.8, 9.9, 9.10, 9.11, 9.12, 9.13],
            [10.6, 10.7, 10.8, nan,  nan,  nan,  nan,  nan],
            [11.6, 11.7, 11.8, nan,  nan,  nan,  nan,  nan],
            [12.6, 12.7, 12.8, nan,  nan,  nan,  nan,  nan],
            [ nan, 13.7, 13.8, nan,  nan,  nan,  nan,  nan],
            [ nan, 14.7, 14.8, nan,  nan,  nan,  nan,  nan],
            [ nan, 15.7, 15.8, nan,  nan,  nan,  nan,  nan],
            [ nan, 16.7, 16.8, nan,  nan,  nan,  nan,  nan],
            [ nan, 17.7, 17.8, nan,  nan,  nan,  nan,  nan],
        ])

        self.expected_g1_3 = np.array([
            [ nan,  nan,  1.4, 1.5, 1.6, 1.7, 1.8],
            [ nan,  nan,  2.4, 2.5, 2.6, 2.7, 2.8],
            [ nan,  nan,  3.4, 3.5, 3.6, 3.7, 3.8],
            [ nan,  nan,  4.4, 4.5, 4.6, 4.7, 4.8],
            [ nan,  nan,  5.4, 5.5, 5.6, 5.7, 5.8],
            [ 6.6,  6.7,  6.8, nan, nan, nan, nan],
            [ 7.6,  7.7,  7.8, nan, nan, nan, nan],
            [ 8.6,  8.7,  8.8, nan, nan, nan, nan],
            [ 9.6,  9.7,  9.8, nan, nan, nan, nan],
            [10.6, 10.7, 10.8, nan, nan, nan, nan],
            [11.6, 11.7, 11.8, nan, nan, nan, nan],
            [12.6, 12.7, 12.8, nan, nan, nan, nan],
        ])

        self.expected_g3_4 = np.array([
            [0.0, 0.1, 0.2, 0.3, nan, nan, nan, nan, nan],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
            [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
            [nan, nan, nan, nan, 4.4, 4.5, 4.6, 4.7, 4.8],
            [nan, nan, nan, nan, 5.4, 5.5, 5.6, 5.7, 5.8],
        ])

        self.expected_all_gs = np.array([
            [0.0, 0.1, 0.2, 0.3, nan, nan,  nan,  nan,  nan, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5,  1.6,  1.7,  1.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5,  2.6,  2.7,  2.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [3.0, 3.1, 3.2, 3.3, 3.4, 3.5,  3.6,  3.7,  3.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, 4.4, 4.5,  4.6,  4.7,  4.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, 5.4, 5.5,  5.6,  5.7,  5.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan,  6.6,  6.7,  6.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan,  7.6,  7.7,  7.8, 7.9, 7.10, 7.11, 7.12, 7.13, 7.14, 7.15, 7.16],
            [nan, nan, nan, nan, nan, nan,  8.6,  8.7,  8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14, 8.15, 8.16],
            [nan, nan, nan, nan, nan, nan,  9.6,  9.7,  9.8, 9.9, 9.10, 9.11, 9.12, 9.13,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan, 10.6, 10.7, 10.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan, 11.6, 11.7, 11.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan, 12.6, 12.7, 12.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan,  nan, 13.7, 13.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan,  nan, 14.7, 14.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan,  nan, 15.7, 15.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan,  nan, 16.7, 16.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
            [nan, nan, nan, nan, nan, nan,  nan, 17.7, 17.8, nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
        ])

    def test_vertical_merge_above(self):
        g_merged = misc.padded_stack(self.g1, self.g3, how='v', where='-', shift=2)
        nptest.assert_array_equal(g_merged, self.expected_g1_3)

    def test_vertical_merge_below(self):
        g_merged = misc.padded_stack(self.g3, self.g1, how='v', where='+', shift=-2)
        nptest.assert_array_equal(g_merged, self.expected_g1_3)

    def test_horizontal_merge_left(self):
        g_merged = misc.padded_stack(self.g1, self.g2, how='h', where='-', shift=1)
        nptest.assert_array_equal(g_merged, self.expected_g1_2_left)

    def test_horizontal_merge_right(self):
        g_merged = misc.padded_stack(self.g1, self.g2, how='h', where='+', shift=1)
        nptest.assert_array_equal(g_merged, self.expected_g1_2_right)

    def test_vert_merge_0and1(self):
        merged = misc.padded_stack(self.g0, self.g1, how='v', where='-', shift=1)
        nptest.assert_array_equal(merged, self.expected_g0_1)

    def test_vert_merge_0and1and2(self):
        step1 = misc.padded_stack(self.g0, self.g1, how='v', where='-', shift=-1)
        step2 = misc.padded_stack(step1, self.g2, how='h', where='+', shift=1)
        nptest.assert_array_equal(step2, self.expected_g0_1_2)

    def test_big_grid_lowerleft_to_upperright(self):
        step1 = misc.padded_stack(self.g4, self.g3, how='h', where='+', shift=1)
        step2 = misc.padded_stack(step1, self.g1, how='v', where='+', shift=6)
        step3 = misc.padded_stack(step2, self.g2, how='h', where='+', shift=7)
        step4 = misc.padded_stack(step3, self.g5, how='h', where='+', shift=7)
        step5 = misc.padded_stack(step4, self.g0, how='v', where='+', shift=7)

        nptest.assert_array_equal(step5, self.expected_all_gs)

    def test_big_grid_upperright_to_lowerleft(self):
        step1 = misc.padded_stack(self.g0, self.g1, how='v', where='-', shift=-1)
        step2 = misc.padded_stack(step1, self.g2, how='h', where='+', shift=1)
        step3 = misc.padded_stack(step2, self.g3, how='v', where='-', shift=-2)
        step4 = misc.padded_stack(step3, self.g4, how='h', where='-', shift=-1)
        step5 = misc.padded_stack(step4, self.g5, how='h', where='+', shift=7)

        nptest.assert_array_equal(step5, self.expected_all_gs)


class test__outputfile(object):
    def test_basic(self):
        nt.assert_equal(
            misc._outputfile('this', 'that.txt'),
            os.path.join('this', 'that.txt')
        )

    def test_withNone(self):
        nt.assert_equal(
            misc._outputfile(None, 'that.txt'),
            os.path.join('.', 'that.txt')
        )


class test__NodeSet(object):
    def setup(self):
        from numpy import nan
        self.A = misc._NodeSet(np.arange(12).reshape(4, 3) * 1.0)
        self.B = misc._NodeSet(np.arange(8).reshape(2, 4) * 1.0)
        self.C = misc._NodeSet(np.arange(18).reshape(3, 6) * 1.0)

        self.known_flipped_A_nodes = np.array([
            [ 2.,  1.,  0.],
            [ 5.,  4.,  3.],
            [ 8.,  7.,  6.],
            [11., 10.,  9.]
        ])


        self.known_transposed_transformed_A_nodes = np.array([
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


    def test_nodes_and_setter(self):
        set_val = np.arange(16).reshape(4, 4) * 1
        self.A.nodes = set_val
        nptest.assert_array_equal(set_val, self.A.nodes)

    def test_transform(self):
        nptest.assert_array_equal(
            self.known_flipped_A_nodes,
            self.A.transform(np.fliplr).nodes
        )

    def test_transpose(self):
        nptest.assert_array_equal(
            self.A.nodes.T,
            self.A.transpose().nodes
        )

    def test_transpose_transform(self):
        nptest.assert_array_equal(
            self.known_transposed_transformed_A_nodes,
            self.A.transpose().transform(np.flipud).nodes
        )

    def test_transform_transpose(self):
        nptest.assert_array_equal(
            self.known_transposed_transformed_A_nodes,
            self.A.transpose().transform(np.flipud).nodes
        )

    def test_merge_vplus_0(self):
        nptest.assert_array_equal(
            self.known_AB_vplus_0,
            self.A.merge(self.B, how='v', where='+', shift=0).nodes,
        )

    def test_merge_vminus_0(self):
        nptest.assert_array_equal(
            self.known_AB_vminus_0,
            self.A.merge(self.B, how='v', where='-', shift=0).nodes,
        )

    def test_merge_vplus_2(self):
        nptest.assert_array_equal(
            self.known_AB_vplus_2,
            self.A.merge(self.B, how='v', where='+', shift=2).nodes,
        )

    def test_merge_vminus_2(self):
        nptest.assert_array_equal(
            self.known_AB_vminus_2,
            self.A.merge(self.B, how='v', where='-', shift=2).nodes,
        )

    def test_merge_vplus_neg1(self):
        nptest.assert_array_equal(
            self.known_AB_vplus_neg1,
            self.A.merge(self.B, how='v', where='+', shift=-1).nodes,
        )

    def test_merge_vminus_neg1(self):
        nptest.assert_array_equal(
            self.known_AB_vminus_neg1,
            self.A.merge(self.B, how='v', where='-', shift=-1).nodes,
        )

    def test_merge_hplus_0(self):
        nptest.assert_array_equal(
            self.known_AB_hplus_0,
            self.A.merge(self.B, how='h', where='+', shift=0).nodes,
        )

    def test_merge_hminus_0(self):
        nptest.assert_array_equal(
            self.known_AB_hminus_0,
            self.A.merge(self.B, how='h', where='-', shift=0).nodes,
        )

    def test_merge_hplus_2(self):
        nptest.assert_array_equal(
            self.known_AB_hplus_2,
            self.A.merge(self.B, how='h', where='+', shift=2).nodes,
        )

    def test_merge_hminus_2(self):
        nptest.assert_array_equal(
            self.known_AB_hminus_2,
            self.A.merge(self.B, how='h', where='-', shift=2).nodes,
        )

    def test_merge_hplus_neg1(self):
        nptest.assert_array_equal(
            self.known_AB_hplus_neg1,
            self.A.merge(self.B, how='h', where='+', shift=-1).nodes,
        )

    def test_merge_hminus_neg1(self):
        nptest.assert_array_equal(
            self.known_AB_hminus_neg1,
            self.A.merge(self.B, how='h', where='-', shift=-1).nodes,
        )


class test_ModelGrid(object):
    def setup(self):
        self.x, self.y = testing.make_nodes()
        self.g1 = misc.ModelGrid(self.x[:, :3], self.y[:, :3])
        self.g2 = misc.ModelGrid(self.x[2:5, 3:], self.y[2:5, 3:])

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
        self.known_coord_pairs = np.array([
            [ 1. ,  0. ], [ 1.5,  0. ], [ 2. ,  0. ], [ 1. ,  0.5],
            [ 1.5,  0.5], [ 2. ,  0.5], [ 1. ,  1. ], [ 1.5,  1. ],
            [ 2. ,  1. ], [ 1. ,  1.5], [ 1.5,  1.5], [ 2. ,  1.5],
            [ 1. ,  2. ], [ 1.5,  2. ], [ 2. ,  2. ], [ 1. ,  2.5],
            [ 1.5,  2.5], [ 2. ,  2.5], [ 1. ,  3. ], [ 1.5,  3. ],
            [ 2. ,  3. ], [ 1. ,  3.5], [ 1.5,  3.5], [ 2. ,  3.5],
            [ 1. ,  4. ], [ 1.5,  4. ], [ 2. ,  4. ]
        ])

    def test_nodes_x(self):
        nt.assert_true(hasattr(self.g1, 'nodes_x'))
        nt.assert_true(isinstance(self.g1.nodes_x, misc._NodeSet))

    def test_nodes_y(self):
        nt.assert_true(hasattr(self.g1, 'nodes_y'))
        nt.assert_true(isinstance(self.g1.nodes_y, misc._NodeSet))

    def test_icells(self):
        nt.assert_equal(
            self.g1.icells,
            self.known_cols
        )

    def test_jcells(self):
        nt.assert_equal(
            self.g1.jcells,
            self.known_rows
        )

    def test_template(self):
        nt.assert_equal(self.g1.template, None)

        template_value = 'junk'
        self.g1.template = template_value
        nt.assert_equal(self.g1.template, template_value)

    def test_as_dataframe(self):
        pdtest.assert_frame_equal(
            self.g1.as_dataframe(),
            self.known_df,
            check_names=False
        )

        pdtest.assert_index_equal(
            self.g1.as_dataframe().columns,
            self.known_df.columns,
        )

    def test_as_coord_pairs(self):
        nptest.assert_array_equal(
            self.g1.as_coord_pairs(),
            self.known_coord_pairs
        )

    def test_transform(self):
        gx = self.g1.x.copy() * 10
        g = self.g1.transform(lambda x: x * 10)
        nptest.assert_array_equal(g.x, gx)

    def test_fliplr(self):
        gx = np.fliplr(self.g1.x.copy())
        g = self.g1.fliplr()
        nptest.assert_array_equal(g.x, gx)

    def test_flipud(self):
        gx = np.flipud(self.g1.x.copy())
        g = self.g1.flipud()
        nptest.assert_array_equal(g.x, gx)

    def test_merge(self):
        g3 = self.g1.merge(self.g2, how='horiz', where='+', shift=2)
        g4 = misc.ModelGrid(self.x, self.y)

        nptest.assert_array_equal(g3.x, g4.x)
