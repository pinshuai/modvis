# Usage

Below is a list of functions available in `modvis`.

- To load visdump file:

!!! note
    For subsurface, use `domain=None` because it load subsurface by default. `columnar=True` is required for reordering the cells to column-based for easy plotting. 

```python
import modvis
# load subsurface visfile
modvis.ats_xdmf.VisFile(model_dir='.', domain=None, load_mesh=True, columnar=True)

# load surface visfile
modvis.ats_xdmf.VisFile(model_dir='.', domain='surface', load_mesh=True)

```

- To plot surface ponded depth:

!!! note
    You can use either numbering (0,1,2...) or datetime string (e.g., "2015-10-01") for the `time_slice`.


```python
modvis.plot_vis_file.plot_surface_data(visfile, var_name="surface-ponded_depth",
				time_slice=0)
```

- To plot subsurface saturation at the top layer:

```python
modvis.plot_vis_file.plot_layer_data(visfile, var_name = "saturation_liquid", 
                             layer_ind = 0, time_slice= "2015-10-01")
```

- To plot flow duration curve

```python
modvis.general_plots.plot_FDC(dfs, labels, colors)
```

- To plot one-to-one scatter plot with metrics

```python
modvis.general_plots.one2one_plot(obs_df, simu_df, metrics=['KGE'])
```
