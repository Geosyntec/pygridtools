from importlib import resources

from pygridgen.tests import requires

try:
    import pytest
except ImportError:
    pytest = None


@requires(pytest, 'pytest')
def test(*args):
    options = [str(resources.files("pygridtools"))]
    options.extend(list(args))
    return pytest.main(options)


@requires(pytest, 'pytest')
def teststrict(*args):
    options = list(set([
        str(resources.files("pygridtools")),
        '--mpl',
        '--doctest-modules'
    ] + list(args)))
    return pytest.main(options)
