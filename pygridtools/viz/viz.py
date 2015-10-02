from . import _viz_mpl
from . import _viz_bokeh


def _check_engine(engine):
    mpl_strings = ['matplotlib', 'mpl', 'seaborn', 'sns']
    bokeh_strings = ['bokeh', 'bp']
    if engine.lower() in mpl_strings:
        return _viz_mpl
    elif engine.lower() in bokeh_strings:
        return _viz_bokeh
    else:
        raise ValueError("'{}' is not a valid engine".format(engine))


def plotDomain(*args, **kwargs):
    engine_module = _check_engine(kwargs.pop('engine', 'mpl'))
    return engine_module._plot_domain(*args, **kwargs)


def plotBoundaries(*args, **kwargs):
    engine_module = _check_engine(kwargs.pop('engine', 'mpl'))
    return engine_module._plot_boundaries(*args, **kwargs)


def plotPoints(*args, **kwargs):
    engine_module = _check_engine(kwargs.pop('engine', 'mpl'))
    return engine_module._plot_points(*args, **kwargs)


def plotCells(*args, **kwargs):
    engine_module = _check_engine(kwargs.pop('engine', 'mpl'))
    return engine_module._plot_cells(*args, **kwargs)

