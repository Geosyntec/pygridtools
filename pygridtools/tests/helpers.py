from pkg_resources import resource_filename
from contextlib import contextmanager
from functools import wraps

import pytest


def raises(error):
    """Wrapper around pytest.raises to support None."""
    if error:
        return pytest.raises(error)
    else:
        @contextmanager
        def not_raises():
            try:
                yield
            except Exception as e:
                raise e
        return not_raises()


def requires(module, modulename):
    def outer_wrapper(function):
        @wraps(function)
        def inner_wrapper(*args, **kwargs):
            if module is None:
                raise RuntimeError("{} required for `{}`".format(modulename, function.__name__))
            else:
                return function(*args, **kwargs)
        return inner_wrapper
    return outer_wrapper
