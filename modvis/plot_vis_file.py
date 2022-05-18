import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import collections as pltc
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
# import sys
# sys.path.append("../")
from modvis.ATSutils import rmLeapDays
# import fiona, shapely
from datetime import datetime

rho_m = 55500 # moles/m^3, water molar density
rho = 997
g = 9.80665
atm_p = 101325

def get_vertex_value_from_cell(vertex_xyz, iconn, idat):
    """get vertex value from cell value. This is useful for contour plot."""
    val = []
    for i in np.unique(iconn):
        irow, icol = np.where(iconn == i)   
        ival = idat[irow].mean() 
        val.append(ival)  
        
    return val

def map_subsurface2surface(sub_map, sub_centroids, surface_centroids):
    """map cells from subsurface to surface."""
    top_cells = sub_map[:, -1].flatten()
    sub_top_centr = sub_centroids[top_cells, :]
    subsurface2surface = []
    for ixyz in surface_centroids:
        idx = np.where((sub_top_centr[:, 0] == ixyz[0]) &  (sub_top_centr[:, 1] == ixyz[1]))[0][0]
        subsurface2surface.append(idx)
        
    return subsurface2surface

def map_surface2subsurface(sub_map, sub_centroids, surface_centroids):
    """map cells from surface to subsurface."""
    top_cells = sub_map[:, -1].flatten()
    sub_top_centr = sub_centroids[top_cells, :]
    surface2subsurface = []
    for ixyz in sub_top_centr:
        idx = np.where((surface_centroids[:, 0] == ixyz[0]) &  (surface_centroids[:, 1] == ixyz[1]))[0][0]
        surface2subsurface.append(idx)
        
    return surface2subsurface

def get_time(vis_data, time_slice=None, **kwargs):
    """get time index if time_slice is str formatted"""
    times = vis_data.times
    times = rmLeapDays(times, **kwargs)
    
    if time_slice is not None:
        try:
            time_idx = int(time_slice)
        except ValueError:
            itime = datetime.strptime(time_slice, '%Y-%m-%d')
            time_idx = np.where(times == itime)[0][0] 
        return times, time_idx  
    else:
        return times

def plot_water_content(vis_data, 
                       origin_date = '1980-01-01', layer_ind=0, time_slice = -1, colorbar = False, ax = None, title = None,  **kwargs):
    """plot water content in a single layer in the subsurface.
    Parameters:
        vis_data, ats_xdmf.VisFile object
        origin_date, str, default to 1980-01-01
            origin_date of the model
        layer_ind, int, 0-indexed
            layer id with 0 being on top and -1 on bottom. Note the actual layers are ordered from bottom up.
        time_slice, int, 0-indexed
            time index-0,1,...,-1
        colorbar, bool
            whether to add colorbar
        ax, axis handel. Default is creating one.
        
    Returns:
        fig, ax   
    """
    ordered_centroids = vis_data.centroids
    vertex_xyz = vis_data.vertex_xyz
    conn = vis_data.conn
    map = vis_data.map

    times = vis_data.times
    datetime = rmLeapDays(times, origin_date = origin_date)
    
    sat = vis_data.getArray("saturation_liquid.cell.0")
    por = vis_data.getArray("porosity.cell.0")
    
    # layer ordered from bottom to top
    layers = np.arange(map.shape[-1])[::-1]
    ilayer = layers[layer_ind]
#     icoord = ordered_centroids[:, ilayer, :]
    icells = map[:, ilayer].flatten()
    isat = sat[time_slice, :, ilayer]
    ipor = por[time_slice, :, ilayer]
    idat = isat*ipor
    
    iconn = conn[icells, -3:]
    
#     vmin = np.nanmin(idat), vmax = np.nanmax(idat)

    if ax == None:
        fig, ax = plt.subplots(1,1, figsize=(8, 4))    
    ax.set_aspect('equal')
    tpc = ax.tripcolor(vertex_xyz[:,0], vertex_xyz[:,1], iconn, facecolors= idat, edgecolors = 'w', linewidth=0.01,  **kwargs)
    
    if title is None:
        ax.set_title(f"Time: {datetime[time_slice].date()}; Layer: {layer_ind+1}")
    else:
        ax.set_title(title)
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    if colorbar == True:
        clabel = "Water content [-]"
        cb = plt.colorbar(tpc)
        cb.ax.set_ylabel(clabel, labelpad=0.3)
    
    try: 
        return fig, ax, tpc
    except:
        return ax, tpc

def plot_column_head(vis_data, 
                     origin_date='1980-01-01', col_ind = 0, plot = True,
                     ax=None):
    """plot variable in a single column over time in the subsurface.
    Parameters:
        vis_data, object from xdmf.VisFile()
        origin_date, str
            model start origin time. Defaults to 1980-1-1
        col_ind, int
            column index for plotting head
        plot, bool
            plot the head if true
    Returns:
        df, dataframe with head 
    """
    ordered_centroids = vis_data.centroids
    # vertex_xyz = vis_data.vertex_xyz
    # conn = vis_data.conn
    map = vis_data.map
    times = get_time(vis_data, origin_date=origin_date)
    # times = vis_data.times
    # datetime = rmLeapDays(times)
    dat = vis_data.getArray("pressure")

    iz_coord = ordered_centroids[col_ind, :, -1]
    icells = map[col_ind, :].flatten()

    # idat = dat[:, icells]
    idat = dat[:, col_ind, :]
    ih = (idat - atm_p)/rho /g
    # find first saturated cell from bottom up
    try:
        sat_idx = [np.where(ih[i, :] > 0)[0][-1] for i in range(len(times))]
    except:
        unsat_time_id = np.where(ih[:, 0] < 0)[0]
#         print(times[unsat_time_id.tolist()])
        logging.debug(f"water table falls below the bottom at times: {times[unsat_time_id.tolist()][0].date()} d. Use the bottom cell instead.")
        sat_idx = [0]*len(times)

    iH = [ih[i, sat_idx[i]] + iz_coord[sat_idx[i]] for i in range(len(times))]
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8, 3))
        plt.plot(times, iH)
        plt.ylabel('GW table (m)')
        plt.xlabel('')
        plt.title(f'Column: {col_ind}')
        plt.xlim([times.min(), times.max()])
    
    df = pd.DataFrame({"datetime": times, "head [m]": iH})
    df.set_index("datetime", inplace = True)
    return df

def plot_gw_surface(visfile, origin_date="1980-01-01", time_slice = -1, return_head = False, title = True, colorbar = True, contour = False, contourline = True, nlevel = 5, ax = None, **kwargs):
    """Plot groundwater table across the domain at given time. 
    Parameters:
        visfile, ats_xdmf.VisFile object
        origin_date, str
            model start origin time. Defaults to 1980-1-1
        time_slice, int
        return_head, bool
            return groundwater head data if true.
        title, bool
            add datetime as title if true.
        colorbar, bool
            add colorbar if true.
        contour, bool
            If true, plot contourf and contour instead of tripcolor.
        contourline, bool
            add contourline if true.
        nlevel, int
            Number of levels used for contour
        ax, axis
            axis for plotting. Default to create a new one.
    Returns:
        fig, ax
    """
    vertex_xyz = visfile.vertex_xyz
    conn = visfile.conn
    map = visfile.map

    t = visfile.times
    times = rmLeapDays(t)

    try:
        time_idx = int(time_slice)
    except ValueError:
        itime = datetime.strptime(time_slice, '%Y-%m-%d')
        time_idx = np.where(times == itime)[0][0] 
        
#     datetime = times
    press = visfile.getArray("pressure")
    ipress = press[time_idx, :,:]
    # ipress = transect_data[2, time_idx, :, :]
    # iz_coord = transect_data[1, time_idx, :, :]
    iz_coord = visfile.centroids[:,:,-1]
    ih = (ipress - atm_p)/rho /g

    try:
        # get the last saturated cell index at each column
        sat_idx = [np.where(ih[i, :] > 0)[0][-1] for i in range(ih.shape[0])]
    except:
        unsat_col_id = np.where(ih[:, 0]<0)[0]
        logging.debug(f"water table is below the bottom at columns: {unsat_col_id}. Use the bottom layer instead.")
        sat_idx = [0]*ih.shape[0]

    iH = np.asarray([ih[i, sat_idx[i]] + iz_coord[i, sat_idx[i]] for i in range(ih.shape[0])])

    icells = map[:, -1].flatten()
    iconn = conn[icells, -3:]
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8, 4))    
    ax.set_aspect('equal')

    tpc = ax.tripcolor(vertex_xyz[:,0], vertex_xyz[:,1], iconn, facecolors= iH, edgecolors = 'w',
                       linewidth=0.01, **kwargs)  
    
    if contour: 
        if isinstance(nlevel, int):
            levels = np.linspace(np.floor(iH.min()), np.ceil(iH.max()), nlevel)
        elif isinstance(nlevel, list) or isinstance(nlevel, (np.ndarray, np.generic)):
            levels = nlevel
        else:
            raise RuntimeError("Must provide level info for contour! Can be a list of levels of number of levels")
        
        unique_cells = np.unique(iconn)
        vertex_xyz_2D = vertex_xyz[unique_cells]
        iconn_reorder = np.array([np.where(unique_cells == j)[0][0] for i in iconn for j in i]).reshape(-1, 3)
        # get color value at vertex
        val = get_vertex_value_from_cell(vertex_xyz_2D, iconn_reorder, iH)

        tpc = ax.tricontourf(vertex_xyz_2D[:,0], vertex_xyz_2D[:,1], iconn_reorder, 
                       val, levels = levels, extend = 'both', **kwargs)  
        if contourline:
            ax.tricontour(vertex_xyz_2D[:,0], vertex_xyz_2D[:,1], iconn_reorder, 
                       val, colors = 'k', linewidths = 0.5, levels = levels, extend = 'both', **kwargs)          
        
    if colorbar:
        clabel = 'GW table [m]'
        if contour:
            cb = plt.colorbar(tpc)
        else:
            cb = plt.colorbar(tpc, extend = "both")
        cb.ax.set_ylabel(clabel, labelpad=0.3)   
#     cb = plt.colorbar(tpc)
    
    if not title:
        titles = ''
    else:
        titles = f"Time: {times[time_idx].date()}"
    
    ax.set_title(titles)
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
#     cb.ax.set_ylabel("GW table [m]", labelpad=0.4)
    plt.tight_layout()    
    if return_head:
        return iH, ax, tpc
    else:
        try:
            return fig, ax, tpc
        except:
            return ax, tpc

def plot_column_data(vis_data, var_name, origin_date='1980-01-01', col_ind = 0, cmap = None, ylabel = None, plot_contour = False, contour_spacing = 0.01, levels = None, logx = False, ax=None):
    """plot variable in a single column over time in the subsurface.

    Parameters:
        vis_data, object from xdmf.VisFile()
        var_name, str
            variable name in visfile
        origin_date, str
            model start origin time. Defaults to 1980-1-1
        col_ind, int
            column index for plotting head
        cmap, str
            colormap for plotting
        plot_contour, bool
            plot contour if true
        contour_spacing, float
            set spacing if plot_contour is True.
        levels, int
            levels for contours
        logx, bool
            set x-axis to log scale if True

    Returns:
        df, dataframe with head 

    """
    ordered_centroids = vis_data.centroids
    map = vis_data.map

    # times = vis_data.times
    times = get_time(vis_data, origin_date=origin_date)
    dat = vis_data.getArray(var_name)
    
    iz_coord = ordered_centroids[col_ind, :, -1]
    # icells = map[col_ind, :].flatten()
    # idat = dat[:, icells]
    idat = dat[:, col_ind, :]
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8, 3))

    cf = plt.contourf(times,
                       iz_coord,
                       idat.T,
                     cmap= cmap,
                       levels=np.arange(np.round(np.min(idat), 2), np.round(np.max(idat), 2), contour_spacing),
                       extend="both",
                       )
    if plot_contour:
        if levels == None:
            levels = [atm_p]
            # print(levels)
        cc = plt.contour(times,
                           iz_coord,
                           idat.T,
#                          cmap= cmap,
                           levels=levels,
                        colors = 'w'
                           )        
    

    cb = plt.colorbar(cf)
    
    if ylabel == None:
        ylabel = var_name
    cb.ax.set_ylabel(ylabel, labelpad=0.3)
    if logx:
        plt.xscale('log')
    plt.ylabel('Elevevation (m)')
    plt.xlabel('Time (d)')
    plt.title(f'Column: {col_ind}')
    # if logx:
    #     xlim = [None, times.max()]
    # else:
    xlim = [times.min(), times.max()]
    ax.set_xlim(xlim)
    
    try:
        return fig, ax
    except:
        return
def plot_layer_data(vis_data, var_name, origin_date = '1980-01-01', layer_ind = 0, time_slice = -1, cmap = "turbo", colorbar = True, clabel = None, ax = None, log = False, vmin = None, vmax = None, linthresh = 0.01, linscale = 0.01, **kwargs):
    """plot variable in a single layer in the subsurface.
    Parameters:
        vis_data, ats_xdmf.VisFile object
        var_name, str or np.array of data
            variable names in the vis file. e.g., saturation_liquid
        origin_date, str
            original date used in the model
        layer_ind, int, 0-indexed
            layer id with 0 being on top and -1 on bottom. Note the actual layers are ordered from bottom up.
        time_slice, int, 0-indexed
            time index-0,1,...,-1
        cmap, str
            colormap
        colorbar, bool
            whether to add colorbar
        clabel, str
            colorbar label
        ax, axis handel. Default is creating one.
        log, bool
            set plot to log scale if True
        linthresh, float
            keyword for SymLogNorm        
        linscale, float
            see keyword for SymLogNorm
        
    Returns:
        fig, ax   
    """
    ordered_centroids = vis_data.centroids
    vertex_xyz = vis_data.vertex_xyz
    conn = vis_data.conn
    map = vis_data.map
#     times = vis_data.times
#     datetime = rmLeapDays(times)
    times, time_idx = get_time(vis_data, time_slice, origin_date = origin_date)
    
#     dat = vis_data.getArray(var_name)
    if isinstance(var_name, str):
        dat = vis_data.getArray(var_name)
    elif isinstance(var_name, (np.ndarray, np.generic)):
        dat = var_name
        assert(dat.shape[0] == len(times))
    else:
        raise RuntimeError("Must provide string of variable or np.array of data!")
    
    # layer ordered from bottom to top
    layers = np.arange(map.shape[-1])[::-1]
    ilayer = layers[layer_ind]
#     icoord = ordered_centroids[:, ilayer, :]
    icells = map[:, ilayer].flatten()
    # idat = dat[time_idx, icells]
    idat = dat[time_idx, :, ilayer]
    iconn = conn[icells, -3:]
    if vmin is None:
        vmin = np.nanmin(dat)
    if vmax is None: 
        vmax = np.nanmax(dat)
    if ax == None:
        fig, ax = plt.subplots(1,1, figsize=(8, 4))    
    ax.set_aspect('equal')
    if log:
        tpc = ax.tripcolor(vertex_xyz[:,0], vertex_xyz[:,1], iconn, 
                       facecolors= idat, linewidth=0.01, cmap = cmap, 
                       norm=matplotlib.colors.SymLogNorm(linthresh = linthresh, linscale = linscale, vmin=vmin, vmax=vmax),
                       **kwargs)
    else:
        tpc = ax.tripcolor(vertex_xyz[:,0], vertex_xyz[:,1], iconn,
                facecolors= idat, cmap = cmap, linewidth=0.01, 
                vmin = vmin, vmax = vmax, **kwargs)
    
    ax.set_title(f"Time: {times[time_idx].date()}; Layer: {layer_ind+1}")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    if colorbar == True:
        if clabel is None and isinstance(var_name, str):
            clabel = var_name 
        elif clabel is None:
            clabel = ''
#         clabel = var_name
        cb = plt.colorbar(tpc)
        cb.ax.set_ylabel(clabel, labelpad=0.3)
#     plt.tight_layout()
    
    if ax is None:
        return fig, ax, tpc
    else:
        return ax, tpc

def plot_surface_data(vis_data, var_name,
                      origin_date="1980-01-01", time_slice = -1,
                      facecolors = None,  subset= False, subset_idx = None,
                      colorbar = True, clabel = None, title = True, ax = None,
                      log = False, vmin = None, vmax = None, robust = False,
                      contour = False, nlevel = 5, linthresh = 0.01, linscale =
                      0.01, data_lim = None, **kwargs):
    """plot variable on the surface.
    Parameters:
        vis_data, ats_xdmf.VisFile object
        var_name, str or np.array of actual data
            variable names in the vis file. e.g., saturation_liquid
        origin_date, str
            original date set in the model.
        time_slice, int or str with %Y-%m-%d format
            zero-based index or "%Y-%m-%d"
        facecolors, default to None (get from vis_data)
            1D numpy array with shape=ntris
        cmap, str
            colormap from matplotlib, default to viridis
        subset, bool
            whether to subset data or not. If this is True, must also provide subset_idx
        subset_idx, list or array like
            list of index for subsetting.
        colorbar, bool
            whether to add colorbar or not
        clabel, str
            colorbar label
        title, bool
            whether to add title of timestamp or not
        ax, axis
        log, bool
            whether to transform data or not
        vmin, vmax
            colormap vmin and vmax; Default to None
        robust, bool
            whether to use robust color range [2,98] percentile or not
        nlevel, int or list, default to None
            Number of levels used for contour plot or a list of levels.
        data_lim, None or list
            Only show data within this range if provided. Out ranged values are
            marked as Nans. 
    Returns:
        fig, ax   
    """
    vertex_xyz = vis_data.vertex_xyz
    conn = vis_data.conn
    volume = vis_data.volume

    times, time_idx = get_time(vis_data, time_slice, origin_date=origin_date)
    
    if isinstance(var_name, str):
        dat = vis_data.getArray(var_name)
    elif isinstance(var_name, (np.ndarray, np.generic)):
        dat = var_name
        assert(dat.shape[0] == len(times))
    else:
        raise RuntimeError("Must provide string of variable or np.array of data!")
    
    if isinstance(var_name, str):
        if 'transpiration' in var_name or 'precip' in var_name or 'snow-melt' in var_name:
            #convert m/s to mm/d
            dat = dat*86400*1000 
            unit = '[mm/d]'
        elif 'flux' in var_name:
            #convert mol/s to m^3/d
            dat = dat / rho_m *86400 
            dat = dat/ volume *1000 # m3/d to mm/d
            unit = '[mm/d]'
        elif "snow-water_equivalent" in var_name:
            # convert m to mm
            dat = dat*1000 
            unit = '[mm]'
        else:
            logging.info("No unit convertion.")
            unit = ''      
    
    idat = dat[time_idx, :]

    if data_lim is not None:
        if data_lim[0] is not None:
            idat[idat < data_lim[0]] = np.nan
        if data_lim[1] is not None:
            idat[idat > data_lim[1]] = np.nan
        

    iconn = conn[:, -3:]
    
    if robust:
        vmin, vmax = np.nanpercentile(idat, [2, 98])
        
    if facecolors is not None:
        colors = facecolors
    else:
        colors = idat
        
    if subset:
        iconn = iconn[subset_idx]
        colors = colors[subset_idx]
        
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8, 4))
    ax.set_aspect('equal')
    
    if log and not contour: 
        # use SymLogNorm to normalize in both positive and negative directions.
        tpc = ax.tripcolor(vertex_xyz[:,0], vertex_xyz[:,1], iconn, 
                       facecolors= colors, linewidth=0.01,norm=matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax), **kwargs)
    elif not contour: 
        tpc = ax.tripcolor(vertex_xyz[:,0], vertex_xyz[:,1], iconn, 
                       facecolors= colors, linewidth=0.01, vmin=vmin, vmax=vmax, **kwargs)
    elif contour:
        
        if isinstance(nlevel, int):
            levels = np.linspace(np.floor(idat.min()), np.ceil(idat.max()), nlevel)
        elif isinstance(nlevel, list):
            levels = nlevel
        else:
            raise RuntimeError("Must provide level info for contour! Can be a list of levels of number of levels")
        # get color value at vertex
        val = get_vertex_value_from_cell(vertex_xyz, iconn, idat)

        tpc = ax.tricontourf(vertex_xyz[:,0], vertex_xyz[:,1], iconn, 
                       val, levels = levels, extend = 'both', **kwargs)  
        ax.tricontour(vertex_xyz[:,0], vertex_xyz[:,1], iconn, 
                       val, colors = 'k', linewidths = 0.5, levels = levels, extend = 'both', **kwargs)  
   
    
    if title is False:
        titles = ''
    else:
        titles = f"Time: {times[time_idx].date()}"
    ax.set_title(titles)
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    if colorbar == True:
        if clabel is None and isinstance(var_name, str):
            clabel = var_name + ' '+ unit
        cb = plt.colorbar(tpc, 
#                           extend = "both"
                         )
        cb.ax.set_ylabel(clabel, labelpad=5)
    
    try:
        return fig, ax, tpc
    except:
        return ax, tpc
    
# def plot_riverbed(source, vertex_xyz, triangles, rivers=None, dist_to_river = 200, plot = True):
#     """find riverbed region given river shapefile or index."""

#     try:
#         if rivers is None:
#             with fiona.open(source, 'r') as fid:
#                 profile = fid.profile
#                 shps = [s for (i,s) in fid.items()]  
#             comids = [shp['properties']['COMID'] for shp in shps]

#             rivers = [shapely.geometry.shape(shape['geometry']) for shape in shps]

#         river_multiline = shapely.geometry.MultiLineString(rivers)

#         distances = []
#         for tri in triangles:
#             verts = vertex_xyz[tri]
#             bary = np.sum(np.array(verts), axis=0)/3
#             bary_p = shapely.geometry.Point(bary[0], bary[1])
#             distances.append(bary_p.distance(river_multiline))
#         distances = np.array(distances)

#         river_idx = distances < dist_to_river  
#     except:
#         river_idx = np.loadtxt(source, dtype='bool')
          
#     assert(river_idx.shape[0] == triangles.shape[0])
    
#     if plot:
#         fig, ax = plt.subplots(1,1, figsize=(6,4))
#         ax.tripcolor(vertex_xyz[:,0], vertex_xyz[:,1], triangles[river_idx], vertex_xyz[:,2], 
#               edgecolors = 'w', linewidth=0.01)
#         lines = [np.array(l.coords)[:,0:2] for l in rivers]
#         lc = pltc.LineCollection(lines, linewidths = 0.5, color = 'w')
#         res = ax.add_collection(lc)
# #         plt.plot(*river_multiline.exterior.xy)
#         plt.title(f"Distance < {dist_to_river} m")
        
#     try:
#         return river_idx, shps
#     except:
#         return river_idx
