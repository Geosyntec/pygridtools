[![Build Status](https://travis-ci.org/phobson/pygridtools.svg?branch=master)](https://travis-ci.org/phobson/pygridtools)

# pygridtools
Miscellaneous utilities to accompany pygridgen

## Summary
This package aims to provide a convenient interface to
[`pygridgen`](https://github.com/hetland/pygridgen) for
  + grid creation
  + merging grids together
  + attaching bathymetry or model output to grids for...
  + visualizing grids (via [matplotlib](www.matplotlib.org) or 
    [bokeh](http://bokeh.pydata.org/en/latest/))
  
## Installation
Use [`conda`](http://conda.pydata.org/) and grab the dependecies
in my channel via:

```bash
conda create --yes --name=grids python=2.7
source activate grids
conda config --add channels phobson
conda install --yes seaborn bokeh fiona shapely nose csa nn gridutils gridgen pygridgen
```

Then clone this directory and do `pip install .`

## Python >=3.3 support
Working on it. Gotta work out `pygridgen` first.

## Contributing
  1. Feedback is a huge contribution
  2. Get in touch by creating an issue to make sure we 
     don't duplicate work
  3. Fork this repo
  4. Submit a PR in a seperate branch
  5. Write a test (or two (or three))
  6. Stick to PEP8-ish -- I'm lenient on the 80 chars thing.
     (<100 is probably a smart move though)
  
