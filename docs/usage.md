# Usage

Below is a list of functions available in `modvis`.

- To load visdump file:

```python
import modvis
# load subsurface visfile
modvis.ats_xdmf.VisFile(model_dir='.', domain=None, load_mesh=True, columnar=True)

# load surface visfile
modvis.ats_xdmf.VisFile(model_dir='.', domain='surface', load_mesh=True)

```

- To plot surface ponded depth:

```python
modvis.plot_vis_file.plot_surface_data(visfile, var_name="surface-ponded_depth")
```

- To plot subsurface saturation at the top layer:

```python
modvis.plot_vis_file.plot_layer_data(visfile, var_name = "saturation_liquid", 
                             layer_ind = 0, time_slice= "2015-10-01")
```
