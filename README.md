# ModVis


[![image](https://img.shields.io/pypi/v/modvis.svg)](https://pypi.python.org/pypi/modvis)

[![image](https://img.shields.io/conda/vn/conda-forge/modvis.svg)](https://anaconda.org/conda-forge/modvis)


**A python package for model visualization.**


-   Free software: GPL license
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

### Plot variables on triangular meshes:

```python
import modvis.ats_xdmf as xdmf
import modvis.plot_vis_file as pv

# import visdump file
surface_vis = xdmf.VisFile(model_dir='.', domain='surface')
subsurface_vis = xdmf.VisFile(model_dir='.', domain=None, columnar=True)

# plot surface ponded depth
pv.plot_surface_data(surface_vis, var_name="surface-ponded_depth", log=True,
                              time_slice="2019-05-01", vmin=0.01, vmax=4)

# plot subsurface saturation. Note layer index is ordered from top to bottom (0--top).
pv.plot_layer_data(subsurface_vis, var_name = "saturation_liquid", 
                             layer_ind = 0, time_slice= 0,
                              cmap = "coolwarm")
```

### Plot variables on mixed-element meshes:

```python
import modvis.ats_xdmf as xdmf
import modvis.plot_vis_file as pv

# import visdump file
surface_vis = xdmf.VisFile(model_dir='.', domain='surface', mixed_element=True)
subsurface_vis = xdmf.VisFile(model_dir='.', domain=None, mixed_element=True)

# plot surface ponded depth
pv.plot_surface_data(surface_vis, var_name="surface-ponded_depth", 
                              time_slice="2019-05-01", mixed_element=True)

# plot subsurface saturation. Note layer index is ordered from top to bottom (0--top).
pv.plot_layer_data(subsurface_vis, var_name = "saturation_liquid", 
                             layer_ind = 0, time_slice= 0, mixed_element=True)
```

## Examples

Jupyter notebook examples can be found under [examples/notebooks](./examples/notebooks)


## Credits

This work is supported by LDRD funding from PNNL, with continuing support from the Utah Water Research Laboratory.

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
