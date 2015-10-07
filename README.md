[![Build Status](https://travis-ci.org/Geosyntec/pygridtools.svg?branch=master)](https://travis-ci.org/Geosyntec/pygridtools)
[![Coverage Status](https://coveralls.io/repos/Geosyntec/pygridtools/badge.svg?branch=master&service=github)](https://coveralls.io/github/Geosyntec/pygridtools?branch=master)

# pygridtools
Miscellaneous utilities to accompany [`pygridgen`](https://github.com/hetland/pygridgen)

## Summary
This package aims to provide a convenient interface to
[`pygridgen`](https://github.com/hetland/pygridgen) for
  + grid creation
  + merging grids together
  + attaching bathymetry or model output to grids for...
  + visualizing grids (via [matplotlib](www.matplotlib.org) or eventually
    [bokeh](http://bokeh.pydata.org/en/latest/))

## Installation
Use [`conda`](http://conda.pydata.org/) and grab the dependecies
in my channel via:

```bash
conda create --name=grid python=3.4 seaborn fiona nose 
source activate grid
conda install --channel=IOOS pygridgen
```

Then clone this directory and do `pip install .`

## Python versions.
Currently tested on:
  - Legacy Python
  - 3.4

## Examples
  1. [Basic grid generation and focusing](http://nbviewer.ipython.org/github/phobson/pygridtools/blob/master/examples/1%20-%20Gridgen%20Basics.ipynb)
  2. [Masking dry (land) cells and GIS output](http://nbviewer.ipython.org/github/phobson/pygridtools/blob/master/examples/2%20-%20Shapefiles%20and%20masking%20cells.ipynb)
  3. [Merging grids together](http://nbviewer.ipython.org/github/Geosyntec/pygridtools/blob/master/examples/3%20-%20Merging%20Grids.ipynb)


## Contributing
  1. Feedback is a huge contribution
  2. Get in touch by creating an issue to make sure we
     don't duplicate work
  3. Fork this repo
  4. Submit a PR in a seperate branch
  5. Write a test (or two (or three))
  6. Stick to PEP8-ish -- I'm lenient on the 80 chars thing.
     (<100 is probably a smart move though)

