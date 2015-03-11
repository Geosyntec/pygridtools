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


class test_Grid(object):
    def setup(self):
        self.grid = testing.makeSimpleGrid()
        self.gdf = testing.makeSimpleGrid(as_gridgen=False)
        self.other_gdf = testing.makeSimpleGrid(as_gridgen=False)
        self.top_col_level = ['northing', 'easting']
        self.df_attrs = ['u', 'v', 'centers', 'psi', 'nodes']
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
        nt.assert_equal(None, self.gdf.transform)
        #nt.assert_equal(self.grid, self.gdf.grid)
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

    def test_writeToShapefile_Point(self):
        self.gdf.writeToShapefile(
            self.point_output,
            geomtype='Point',
            template=self.template
        )

        testing.compareShapefiles(
            self.point_output, self.point_baseline, atol=0.1
        )

    def test_writeToShapefile_Polygon(self):
        self.gdf.writeToShapefile(
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
        pdtest.assert_frame_equal(df_merged, self.expected_df1and3)

    def test_vertical_merge_below(self):
        df_merged = misc.mergePoints(self.df3, self.df1, how='j', where='+', offset=2)
        pdtest.assert_frame_equal(df_merged, self.expected_df1and3)

    def test_horizontal_merge_left(self):
        df_merged = misc.mergePoints(self.df2, self.df1, how='i', where='-', offset=-1)
        pdtest.assert_frame_equal(df_merged, self.expected_df1and2)

    def test_horizontal_merge_right(self):
        df_merged = misc.mergePoints(self.df1, self.df2, how='i', where='+', offset=1)
        pdtest.assert_frame_equal(df_merged, self.expected_df1and2)

    def test_vert_merge_0and1(self):
        merged = misc.mergePoints(self.df0, self.df1, how='j', where='-', offset=-1)
        pdtest.assert_frame_equal(merged, self.expected_df0and1)

    def test_vert_merge_0and1and2(self):
        step1 = misc.mergePoints(self.df0, self.df1, how='j', where='-', offset=-1)
        step2 = misc.mergePoints(step1, self.df2, how='i', where='+', offset=1)
        pdtest.assert_frame_equal(step2, self.expected_df0and1and2)

    def test_big_grid_lowerleft_to_upperright(self):
        step1 = misc.mergePoints(self.df4, self.df3, how='i', where='+', offset=1)
        step2 = misc.mergePoints(step1, self.df1, how='j', where='+', offset=6)
        step3 = misc.mergePoints(step2, self.df2, how='i', where='+', offset=7)
        step4 = misc.mergePoints(step3, self.df5, how='i', where='+', offset=7)
        step5 = misc.mergePoints(step4, self.df0, how='j', where='+', offset=7)

        pdtest.assert_frame_equal(step5, self.expected_all_dfs)

    def test_big_grid_upperright_to_lowerleft(self):
        step1 = misc.mergePoints(self.df0, self.df1, how='j', where='-', offset=-1)
        step2 = misc.mergePoints(step1, self.df2, how='i', where='+', offset=1)
        step3 = misc.mergePoints(step2, self.df3, how='j', where='-', offset=-2)
        step4 = misc.mergePoints(step3, self.df4, how='i', where='-', offset=-1)
        step5 = misc.mergePoints(step4, self.df5, how='i', where='+', offset=7)

        pdtest.assert_frame_equal(step5, self.expected_all_dfs)


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
