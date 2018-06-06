import numpy
from matplotlib import pyplot

import pytest
import numpy.testing as nptest


from pygridtools import validate
from pygridtools import testing


def test_mpl_ax_invalid():
    with testing.raises(ValueError):
        validate.mpl_ax('junk')


def test_mpl_ax_with_ax():
    fig, ax = pyplot.subplots()
    fig1, ax1 = validate.mpl_ax(ax)
    assert isinstance(ax1, pyplot.Axes)
    assert isinstance(fig1, pyplot.Figure)
    assert ax1 is ax
    assert fig1 is fig


def test_mpl_ax_with_None():
    fig1, ax1 = validate.mpl_ax(None)
    assert isinstance(ax1, pyplot.Axes)
    assert isinstance(fig1, pyplot.Figure)


@pytest.mark.parametrize(('polycoords', 'error'), [
    ([(2, 2), (5, 2), (5, 5), (2, 5)], None),
    ([(2, 2), (5, 2)], ValueError),
    ([(2, 2, 1), (5, 2, 1), (5, 5, 1), (2, 5, 1)], ValueError),
    ([[(2, 2), (5, 2), (5, 5), (2, 5)], [(2, 2), (5, 2), (5, 5), (2, 5)]], ValueError)
])
def Test_polygon(polycoords, error):
    with testing.raises(error):
        poly = validate.polygon(polycoords)
        nptest.assert_array_equal(numpy.array(polycoords), poly)


def test_xy_array_not_as_pairs():
    _x, _y = numpy.mgrid[:9, :9]
    x, y = validate.xy_array(_x, _y, as_pairs=False)
    nptest.assert_array_equal(x, _x)
    nptest.assert_array_equal(y, _y)


def test_xy_array_as_pairs():
    _y, _x = numpy.mgrid[:3, :3]
    pairs = validate.xy_array(_y, _x)

    known_pairs = numpy.array([
        [0, 0], [0, 1], [0, 2],
        [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2]
    ])
    nptest.assert_array_equal(pairs, known_pairs)


def test_xy_array_diff_shapes():
    with testing.raises(ValueError):
        validate.xy_array(numpy.zeros((3, 3)), numpy.zeros((4, 4)))


def test_xy_array_diff_masks():
    _y, _x = numpy.mgrid[:3, :3]
    mask1 = numpy.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])

    mask2 = numpy.array([
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 0],
    ])

    y = numpy.ma.MaskedArray(data=_y, mask=mask1)
    x = numpy.ma.MaskedArray(data=_x, mask=mask2)
    with testing.raises(ValueError):
        validate.xy_array(x, y)


def test_xy_array_only_one_mask():
    mask1 = numpy.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])

    _y, _x = numpy.mgrid[:3, :3]
    y = numpy.ma.MaskedArray(data=_y, mask=mask1)

    with testing.raises(ValueError):
        validate.xy_array(_x, y)


@pytest.mark.parametrize(('mode', 'expected', 'error'), [
    ('z', None, ValueError),
    ('A', 'a', None),
    ('a', 'a', None),
    ('W', 'w', None),
    ('w', 'w', None)
])
def test_file_mode(mode, expected, error):
    with testing.raises(error):
        result = validate.file_mode(mode)
        assert expected == result


@pytest.mark.parametrize(('x', 'y', 'offset', 'expected'), [
    (numpy.zeros((8, 7)), None, 0, None),
    (numpy.zeros((8, 7)), numpy.zeros((6, 5)), 0, None),
    (numpy.zeros((8, 7)), numpy.ones((8, 7)), 0, numpy.ones((8, 7))),
    (numpy.zeros((8, 7)), numpy.ones((6, 5)), 2, numpy.ones((6, 5))),
])
def test_elev_or_mask(x, y, offset, expected):
    if expected is None:
        with testing.raises(ValueError):
            validate.elev_or_mask(x, y, failNone=True)
    else:
        result = validate.elev_or_mask(x, y, offset=offset)
        nptest.assert_array_equal(result, expected)


def test_equivalent_masks():
    from numpy import nan
    X = numpy.array([
        1, 2, 3, nan, nan, 7,
        1, 2, 3, nan, nan, 7,
        1, 2, 3, nan, nan, nan,
        1, 2, 3, nan, nan, nan,
        1, 2, 3, nan, nan, 7,
    ])

    Y1 = X.copy()

    Y2 = numpy.array([
        1, 2, 3, nan, nan, 7,
        1, 2, 3, nan, nan, nan,
        1, 2, 3, nan, nan, nan,
        1, 2, 3, nan, nan, nan,
        1, 2, 3, nan, nan, 7,
    ])
    with testing.raises(ValueError):
        validate.equivalent_masks(X, Y2)

    x, y = validate.equivalent_masks(X, Y1)
    nptest.assert_array_equal(X, x.data)
    nptest.assert_array_equal(Y1, y.data)
