from pkg_resources import resource_filename

import pygridtools


def test(*args, **kwargs):
    try:
        import pytest
    except ImportError:
        raise ImportError("pytest is requires to run tests")

    alltests = kwargs.pop('alltests', True)
    if alltests:
        options = [resource_filename('pygridtools', '')]
    else:
        options = []

    options.extend(list(args))
    return pytest.main(options)
