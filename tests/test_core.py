import os
import warnings

import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import pandas

import nose.tools as nt
import numpy.testing as nptest
import pandas.util.testing as pdtest

from pygridtools import core
import testing

try:
    import pygridgen
    has_pgg = True
except ImportError:
    has_pgg = False


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
        self.mg = core.ModelGrid(self.xn, self.yn)
        self.g1 = core.ModelGrid(self.xn[:, :3], self.yn[:, :3])
        self.g2 = core.ModelGrid(self.xn[2:5, 3:], self.yn[2:5, 3:])

        self.template = 'tests/test_data/schema_template.shp'
        self.g1.template = self.template
        self.g2.template = self.template

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
        self.known_df.columns.names = ['coord', 'i']
        self.known_df.index.names = ['j']

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
        self.known_masked_cell_df.columns.names = ['coord', 'i']
        self.known_masked_cell_df.index.names = ['j']


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
        nt.assert_equal(self.g1.template, self.template)

        template_value = 'junk'
        self.g1.template = template_value
        nt.assert_equal(self.g1.template, template_value)

        self.g1.template = self.template

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

    @nt.raises(ValueError)
    def test_to_shapefile_bad_geom(self):
        self.g1.to_shapefile('junk', geom='Line')

    def test_to_shapefile_nomask_nodes_points(self):
        outfile = 'tests/result_files/mgshp_nomask_nodes_points.shp'
        basefile = 'tests/baseline_files/mgshp_nomask_nodes_points.shp'
        self.g1.to_shapefile(outfile, usemask=False, which='nodes',
                             geom='point')
        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_nomask_cells_points(self):
        outfile = 'tests/result_files/mgshp_nomask_cells_points.shp'
        basefile = 'tests/baseline_files/mgshp_nomask_cells_points.shp'
        self.g1.to_shapefile(outfile, usemask=False, which='cells',
                             geom='point')
        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_nomask_nodes_polys(self):
        outfile = 'tests/result_files/mgshp_nomask_nodes_polys.shp'
        basefile = 'tests/baseline_files/mgshp_nomask_cells_polys.shp'
        self.g1.to_shapefile(outfile, usemask=False, which='nodes',
                             geom='polygon')
        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_nomask_cells_polys(self):
        outfile = 'tests/result_files/mgshp_nomask_cells_polys.shp'
        basefile = 'tests/baseline_files/mgshp_nomask_cells_polys.shp'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.g1.to_shapefile(outfile, usemask=False, which='cells',
                                 geom='polygon')
            nt.assert_equal(len(w), 1)

        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_mask_cells_points(self):
        outfile = 'tests/result_files/mgshp_mask_cells_points.shp'
        basefile = 'tests/baseline_files/mgshp_mask_cells_points.shp'
        self.g1.cell_mask = self.known_mask

        self.g1.to_shapefile(outfile, usemask=True, which='cells',
                             geom='point')
        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_mask_cells_polys(self):
        outfile = 'tests/result_files/mgshp_mask_cells_polys.shp'
        basefile = 'tests/baseline_files/mgshp_mask_cells_polys.shp'
        self.g1.cell_mask = self.known_mask

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.g1.to_shapefile(outfile, usemask=True, which='cells',
                                 geom='polygon')
            nt.assert_equal(len(w), 1)

        testing.compareShapefiles(outfile, basefile)

    @nt.raises(ValueError)
    def test_to_shapefile_mask_nodes(self):
        self.g1.to_shapefile('junk', usemask=True, which='nodes', geom='point')

    @nt.raises(ValueError)
    def test__get_x_y_nodes_and_mask(self):
        self.g1._get_x_y('nodes', usemask=True)

    @nt.raises(ValueError)
    def test__get_x_y_bad_value(self):
        self.g1._get_x_y('junk', usemask=True)

    def test__get_x_y_nodes(self):
        x, y = self.g1._get_x_y('nodes', usemask=False)
        nptest.assert_array_equal(x, self.g1.xn)
        nptest.assert_array_equal(y, self.g1.yn)

    def test__get_x_y_cells(self):
        x, y = self.g1._get_x_y('cells', usemask=False)
        nptest.assert_array_equal(x, self.g1.xc)
        nptest.assert_array_equal(y, self.g1.yc)

    def test_writeGEFDCControlFile(self):
        known_filename = 'tests/baseline_files/modelgrid_gefdc.inp'
        result_path = 'tests/result_files'
        result_file = 'modelgrid_gefdc.inp'
        self.mg.writeGEFDCControlFile(
            outputdir=result_path,
            filename=result_file,
            title='Model Grid Test'
        )
        testing.compareTextFiles(
            os.path.join(result_path, result_file),
            known_filename
        )

    def test_writeGEFDCCellFile(self):
        known_filename = 'tests/baseline_files/modelgrid_cell.inp'
        result_path = 'tests/result_files'
        result_file = 'modelgrid_cell.inp'
        self.mg.writeGEFDCCellFile(
            outputdir=result_path,
            filename=result_file,
        )
        testing.compareTextFiles(
            os.path.join(result_path, result_file),
            known_filename
        )

    def test_writeGEFDCGridFile(self):
        known_filename = 'tests/baseline_files/modelgrid_grid.out'
        result_path = 'tests/result_files'
        result_file = 'modelgrid_grid.out'
        self.mg.writeGEFDCGridFile(
            outputdir=result_path,
            filename=result_file,
        )
        testing.compareTextFiles(
            os.path.join(result_path, result_file),
            known_filename
        )

    def test_writeGEFDCGridextFiles(self):
        known_filename = 'tests/baseline_files/modelgrid_gridext.inp'
        result_path = 'tests/result_files'
        result_file = 'modelgrid_gridext.inp'
        self.mg.writeGEFDCGridextFile(
            outputdir=result_path,
            filename=result_file,
        )
        testing.compareTextFiles(
            os.path.join(result_path, result_file),
            known_filename
        )

    @nptest.dec.skipif(True)
    def test_plotCells_basic(self):
        fig, ax = self.mg.plotCells()

    @nptest.dec.skipif(True)
    def test_plotCells_boundary(self):
        fig, ax = self.mg.plotCells(
            boundary=testing.makeSimpleBoundary(),
            usemask=True
        )


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

    @nptest.dec.skipif(not has_pgg)
    def test_with_coords_and_bathy(self):
        grid = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy.dropna(),
            **self.gridparams
        )
        nt.assert_true(isinstance(grid, pygridgen.Gridgen))

    @nptest.dec.skipif(not has_pgg)
    def test_with_coords_and_bathy_verbose(self):
        params = self.gridparams.copy()
        params['verbose'] = True
        grid = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy.dropna(),
            **self.gridparams
        )
        nt.assert_true(isinstance(grid, pygridgen.Gridgen))

    @nptest.dec.skipif(not has_pgg)
    @nt.raises(ValueError)
    def test_makegrid_no_nx(self):
        nx = self.gridparams.pop('nx')
        grid = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            **self.gridparams
        )

    @nptest.dec.skipif(not has_pgg)
    @nt.raises(ValueError)
    def test_makegrid_no_ny(self):
        ny = self.gridparams.pop('ny')
        grid = core.makeGrid(
            coords=self.coords,
            bathydata=self.bathy,
            **self.gridparams
        )

