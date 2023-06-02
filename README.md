# ModVis


[![image](https://img.shields.io/pypi/v/modvis.svg)](https://pypi.python.org/pypi/modvis)

[![image](https://img.shields.io/conda/vn/conda-forge/modvis.svg)](https://anaconda.org/conda-forge/modvis)


**A python package for model visualization.**


-   Free software: MIT license
-   Documentation: https://pinshuai.github.io/modvis
    

## Features

-   Visualize model outputs from hydrologic models including ATS, PFLOTRAN, and etc.
-   Plot unstructured meshes with variables such as groundwater table, saturation, and evapotranspiration.
-   Evaluate model performance using different metrics.

## Installation

`ModVis` is available on PyPI. To install, run the following command:

```
pip install modvis
```

If you want to run the latest version of the code, you can install from git:

```
pip install -U git+https://github.com/pinshuai/modvis.git
```

Alternatively, if you want to debug and test the code, you can clone the repository and install from source:

```
git clone https://github.com/pinshuai/modvis.git
cd modvis
pip install -e .
```

## Quick start

To plot variables on the surface mesh:

```python
import modvis.ats_xdmf as xdmf
import modvis.plot_vis_file as pv

# import visdump file
visfile = xdmf.VisFile(model_dir='.', domain='surface', load_mesh=True)

# plot surface ponded depth
pv.plot_surface_data(visfile, var_name="surface-ponded_depth", log=True,
                              time_slice="2019-05-01", vmin=0.01, vmax=4)
```

## Examples

Jupyter notebook examples can be found under [examples/notebooks](./examples/notebooks)


## Credits

This work is supported by LDRD funding from PNNL.

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
