import filecmp


try:
    import pytest
except ImportError:
    pytest = None


import pandas.testing as pdtest

import geopandas


def assert_textfiles_equal(baselinefile, outputfile):
    assert filecmp.cmp(baselinefile, outputfile)


def _standardize_ints(df: geopandas.GeoDataFrame):
    int_cols = df.select_dtypes(include="int")
    for ic in int_cols:
        df[ic] = df[ic].astype("int64")
    return df


def assert_gis_files_equal(baselinefile, outputfile, atol=0.001):
    expected = geopandas.read_file(baselinefile).pipe(_standardize_ints)
    result = geopandas.read_file(outputfile).pipe(_standardize_ints)
    assert_gdfs_equal(expected, result)


def assert_gdfs_equal(expected_gdf: geopandas.GeoDataFrame, result_gdf: geopandas.GeoDataFrame):
    pdtest.assert_frame_equal(
        expected_gdf.drop('geometry', axis=1).sort_index(axis='columns'),
        result_gdf.drop('geometry', axis=1).sort_index(axis='columns'),
        check_column_type=False, check_index_type=False
    )
    assert expected_gdf.geom_almost_equals(result_gdf).all()
