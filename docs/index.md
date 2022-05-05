# Welcome to ModVis


[![image](https://img.shields.io/pypi/v/modvis.svg)](https://pypi.python.org/pypi/modvis)


**A python package for model visualization.**


-   Free software: MIT license
-   Documentation: <https://pinshuai.github.io/modvis>
    

## Features

-   Visualize model outputs from hydrologic models including ATS, PFLOTRAN, and etc.
-   Plot unstructured meshes with variables such as groundwater table, saturation, and evapotranspiration.
-   Evaluate model performance using different metrics.

## Installation

`ModVis` is available on PyPI. To install, run the following command:

```
pip install modvis
```

## Quick start

To plot variables on the surface mesh:

```python
import modvis.plot_vis_file as pv

pv.plot_surface_data(surface_vis, surface_vertex_xyz, surface_conn,
                               var_name="surface-precipitation_rain", log = False,
                              time_slice= "2019-05-01", vmax = 4)
```

## Examples

Jupyter notebook examples can be found in the repo.


## Credits

This work is supported by LDRD funding from PNNL.

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
