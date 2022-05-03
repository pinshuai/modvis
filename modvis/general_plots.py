'''
General plotting functions.
''' 
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np, pandas as pd
import seaborn as sns
import scipy
import logging
import rasterio, fiona
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
import sys
sys.path.append("../myfunctions")
import myfunctions.utils as utils

def gantt_plot(ranges, **kwargs):
    """Plot Gantt plot given start and end date.
    Parameters:
        ranges, DataFrame
        Dataframe with index as name/id, two separate columns ['start', 'end']
    
    """
    fig,ax = plt.subplots(1,1, figsize=(8,6))

    ax = ax.xaxis_date()
    ax = plt.hlines(ranges.index, 
                    mdates.date2num(ranges['start']), 
                    mdates.date2num(ranges['end']), **kwargs)
    fig.tight_layout()  

def plot_FDC(dfs, labels, colors, linestyles=None, start_date=None, end_date=None,
             time_index=None, rank_method = 'average', var=None, ax=None, **kwargs):
    """plot flow duration curve for discharge time series.
    dfs, labels, colors, linestyles must be in list. This works for single dataframe as well.
    See https://streamflow.engr.oregonstate.edu/analysis/flow/index.htm for FDC equations.
    Part of the code is stolen from http://earthpy.org/flow.html.
    Parameters:
        dfs, list
            List of dataframe time series, index must be in datetime
        labels, list
            List of labels for legend
        colors, list
            List of colors for plotting
        linestypes, list
            List of line types for plottong
        start_date, str
            Starting date that pd.index can recognize, e.g., "2015-10-01"
        end_date, str
            Ending date that pd.index can recognize, e.g., "2018-10-01"
        time_index, list
            List of DatetimeIndex for subsetting
        rank_method, str
            Ranking method used by scipy.stats.rankdata. Options include: 'average', 'min','max',
            'dense','ordinal'
        var, str
            Column variable if provided
        ax, axis
            axis for plotting
    """
    ndf = len(dfs)

    if linestyles is None:
        linestyles = ['-']*ndf

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        new_ax = True
    else:
        new_ax = False
        
    if var is not None and isinstance(dfs[0], pd.DataFrame):
        dfs = [df[var] for df in dfs]
    if start_date is not None or end_date is not None:
        dfs = [df.loc[start_date:end_date] for df in dfs]
    if time_index is not None:
        dfs = [df.loc[time_index] for df in dfs]
    # print(dfs)
    df_concat = pd.concat(dfs,axis=1, keys=range(ndf))
    df_concat.dropna(inplace=True)
    # print(df_concat.head)
    # dfs = [df_concat[col] for col in df_concat.columns]
    # print(df_concat.columns)
    # for i,df in enumerate(dfs):
    for i in range(ndf):
        df = df_concat[i]
        # if var is not None and isinstance(df, pd.DataFrame):
        #     df = df[var]
        # if start_date is not None or end_date is not None:
        #     df = df.loc[start_date:end_date]
        # if time_index is not None:
        #     df = df.loc[time_index]
        assert df.isnull().values.any() == False, "nan value exist in df, try remove it!"
        # df.dropna(inplace=True)
        data = df.values
        N = len(data)
        sorted_array = np.sort(data) 
        # calculate the ranks
        ranks = scipy.stats.rankdata(sorted_array, method=rank_method)
        # ranks = ranks[::-1]
        prob = [100*(ranks[i]/(N + 1)) for i in range(N)]     
        
        # Reverse the sorted array so that high flows rank before low flows
        reverse_array = sorted_array[::-1]  
        # prob = np.arange(1,N+1)/(N+1) # this is the same as rankdata(method="ordinal")
        
        ax.plot(prob, reverse_array, linestyles[i], color = colors[i], label = labels[i], **kwargs)
        ax.set_ylabel('Discharge [$m^3/d$]', fontsize = 14)
        ax.set_xlabel('Exceedance probability [%]', fontsize = 14)
        ax.set_yscale('log')
        ax.set_xlim([0, 100])
    plt.grid(linestyle = '-.')
    plt.legend(frameon = True, facecolor = 'w', edgecolor = 'w', framealpha = 0.7) 
    
    if new_ax is True:
        return fig, ax

def plot_raster_on_shape(raster_file, shapefile = None, crop = False, ax = None, colorbar = True, clabel = '', vmin = None, vmax = None, robust = False,  **kwargs):
    """plot raster(.tif, ) and crop based on shapefile.
    Parameters:
        raster_file: str
            path to tiff file
        shapefile: str, optional
            path to shapefile
        crop: bool
            If true, crop raster to shape file
    """
    
    with rasterio.open(raster_file, 'r') as fid:
        profile = fid.profile
        out_image = fid.read(1) # read the first band
        out_image[np.where(out_image == profile['nodata'])] = np.nan # make missing values nan so that it wont show up in the plot
        
    if shapefile is not None:
        with fiona.open(shapefile, "r") as fid:
            shapes = [feature["geometry"] for feature in fid]  
        watershed_shape = gpd.read_file(shapefile)
    
    if shapefile is not None and crop is True:
        with rasterio.open(raster_file, 'r') as fid:
            profile = fid.profile
            out_image, out_transform = rasterio.mask.mask(fid, shapes, crop=True)
            profile.update({ "height" : out_image.shape[1],
                                 "width" : out_image.shape[2],
                                 "transform" : out_transform})  
            out_image = out_image[0, :,:] # get first band
            out_image[np.where(out_image == profile['nodata'])] = np.nan # make missing values nan so that it wont show up in the plot
            
    assert(len(out_image.shape) == 2) # make sure it is 2D array
    
    # note the bound is from upper left to lower right
    x0, y1= profile['transform'] * (0,0)
    x1, y0 = profile['transform'] * (profile['width'], profile['height'])
    extent = x0, x1, y0, y1

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6,5))
    
    if robust:
        vmin, vmax = np.nanpercentile(out_image, [2, 98])

    p = ax.matshow(out_image, extent=extent, vmin = vmin, vmax = vmax, **kwargs)
    if shapefile is not None:
        watershed_shape.boundary.plot(color ='gray',  lw = 0.5,  ax=ax)
    if colorbar:
        cb = plt.colorbar(p, extend = "both", fraction=0.03, pad=0.04)
        cb.ax.set_ylabel(clabel, labelpad=0.3)        
    
    if ax is None:
        return fig, ax

def corrMatrix_plot(df, **kwargs):
    """plot correlation matrix between variables. 
    See this post: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    """
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 5))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot= True, 
                square=True, linewidths=.5, 
                cbar_kws={"shrink": 0.7}, **kwargs
               )    
    plt.title('Correlation Matrix', fontsize = 14)
def varsComp_plot(dfs, variables, labels, colors, linestyles=None, axes=None, **kwargs):
    """compare variables across different model runs"""
    nvar = len(variables)
    if axes is None:
        fig, axes = plt.subplots(nvar,1, figsize=(8,3*nvar), sharex = True)
        new_axes = True
    else:
        new_axes = False
    if linestyles is None:
        linestyles = ['-']*len(dfs)

    if nvar > 1:
        for ax, var in zip(axes, variables):
            for i in np.arange(len(dfs)):
                ax.plot(var, linestyles[i], color = colors[i], data = dfs[i], label = labels[i], **kwargs)
                ax.set_ylabel(var, fontsize = 12)
                ax.set_xlabel('')
                ax.legend(frameon = True, facecolor = 'w', edgecolor = 'w', framealpha = 0.7)
    else:
        ax = axes
        var = variables[0]
        for i in np.arange(len(dfs)):
            ax.plot(var, linestyles[i], color = colors[i], data = dfs[i], label = labels[i], **kwargs)
            ax.set_ylabel(var, fontsize = 12)
            ax.set_xlabel('')
            ax.legend(frameon = True, facecolor = 'w', edgecolor = 'w', framealpha = 0.7)

    if new_axes is True:
        return fig, axes

def make_colormap(colors):
    """make colormap based on discrete colors.
    Parameters:
        colors, list
            A list of matplotlib colors. e.g., ['red', 'green', 'blue']
    """
    return mcolors.LinearSegmentedColormap.from_list("", colors)

def colorbar_index(ncolors, cmap, labels = None):
    """add colorbar index based on colormap and ncolors. This will set ticks in the middle of each color range"""
    if type(cmap) is list:
        cmap = make_colormap(cmap)
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable,fraction=0.03, pad=0.04)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors)) # set tick locations
    # set tick labels
    if labels is not None:
        colorbar.set_ticklabels(labels)
    else:
        colorbar.set_ticklabels(range(ncolors)) 
    
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def colorline(x, y, z = None, norm_lim = None, ax = None, cmap = 'bwr', **kwargs):
    """plot line changing with colormap.
    Parameters:
        x, list or array
        y, list or array
        z, optional
            values that colors mapped to.
        norm_lim, list
            list provided to Normalized()
    Returns:
        line object
    """
    if isinstance(x[0], datetime.date):
        x = mdates.date2num(x)
        
    if z is None:
        z = y
    if norm_lim is None:
        norm = plt.Normalize(vmin=np.nanmin(y), vmax=np.nanmax(y))
    else:
        norm = plt.Normalize(vmin=norm_lim[0], vmax=norm_lim[1])
        
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,4))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # z = np.linspace(0.0, 1.0, len(times))
    lc = LineCollection(segments, array = z, cmap=cmap, norm=norm, **kwargs)
    line = ax.add_collection(lc)

    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    dateFmt = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_formatter(dateFmt)
    # important to set scale! The default is [0,1]
    ax.autoscale_view()
#     plt.colorbar(line, ax=ax) 
    return line
    

def quantile_plot(data, quantiles= [0.25, 0.5, 0.75], axis= 1, arr_index =
                  None, weights = None, ax = None, fill_color = 'slategray',
                  line_color = 'black', **kwargs):
    """Add quantiles to time series plot.
    Parameters:
        data, array like or dataframe
            If it is a dataframe, then index must be datetime format. Columns are variables [y1,y2,y3,...]
        quantiles, list
            Default to [0.25, 0.5, 0.75]. 3 quantiles are required for the plotting.
        axis, int or a tuple
            axis used by np.quantile. Defaults to 1.
        arr_index, list or array like
            Provided if data is array. 
        weights, 
            If weighted, provide a list of weights that has the lenghth of df.shape[1]
            
    Returns:
        time series plot showing quantile interval.
    """
    if isinstance(data, pd.DataFrame):
        arr = data.values
        idx = data.index
    else: 
        arr = data # assume this is numpy.array
        if arr_index is not None:
            assert(len(arr_index) == arr.shape[0])
            idx = arr_index
        else:
            idx = np.arange(arr.shape[0])
    if weights is None:
        quantiles_arr = np.nanquantile(arr, quantiles, axis = axis).T        
    else:
        quantiles_arr = np.vstack([utils.weighted_quantile(arr[i, :], quantiles, sample_weight= weights) for i in range(arr.shape[0])])
    
    # if fill_color is None:
    #     fill_color = 'slategray'
    # if line_color is None:  
    #     line_color = 'slategray'
    cols = [str(int(i*100)) for i in quantiles]
    quantiles_df = pd.DataFrame(quantiles_arr, columns = cols)
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,4))
    ax.fill_between(idx, quantiles_df[cols[0]].values, quantiles_df[cols[-1]].values, alpha = 0.3, color = fill_color, **kwargs)
    ax.plot(idx, quantiles_df[cols[1]].values, lw=0.5, label = 'median',
            color = line_color, **kwargs)
    ax.set_xlim([idx[0], idx[-1]])
    
    return ax, quantiles_df

def one2one_plot(df_obs, df_simu, metrics = ["R^2"], show_metrics=True, ax =
                 None, equal_aspect = False, show_density = False,
                 decompose_KGE = False, **kwargs):
    """One to One plot with a line.
    Parameters:
        df_obs, df_simu are Pandas series.
        metric, list or 'all'
            available metrics are: pearsonr, R2, RMSE, KGE,...
    """
    assert(isinstance(df_obs, pd.Series)) 
    assert(isinstance(df_simu, pd.Series)) 
    if not isinstance(df_obs.index, pd.DatetimeIndex) or not isinstance(df_simu.index, pd.DatetimeIndex):
        raise ValueError("Data Series must have datetime as index.")

    metric_dict, df = utils.get_metrics(df_obs.index, df_obs.values, 
                 df_simu.index, df_simu.values, metrics = metrics)
    if "mKGE" in metrics or 'KGE' in metrics: 
        if decompose_KGE is True:
            KGE_metric, _ = utils.get_metrics(df_obs.index, df_obs.values, 
                 df_simu.index, df_simu.values, metrics = ['mKGE'], return_all
                                         = True)
            print(f"mKGE: {KGE_metric['mKGE'][0]}, cc:  {KGE_metric['mKGE'][1]}, alpha:  {KGE_metric['mKGE'][2]}, beta:  {KGE_metric['mKGE'][3]}")
    
    max_ = max(df['obs'].values.max(), df['simu'].values.max())
    min_ = min(df['obs'].values.min(), df['simu'].values.min())

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,4))
    if equal_aspect:
        ax.set_aspect('equal')   
        
    if show_density:
        x = df['obs'].values
        y = df['simu'].values
        xy = np.vstack([x, y])
        z = scipy.stats.gaussian_kde(xy)(xy)
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, s=50, **kwargs)
    else:
        sns.scatterplot(data = df, x = "obs", y = "simu", ax = ax, **kwargs)
    
    ax.plot([min_/100, max_*100], [min_/100, max_*100], color='darkslategray')
    ax.set_xlim([min_, max_])
    ax.set_ylim([min_, max_])
    
    if show_metrics:
        ct = 0
        for i in metric_dict.keys():
            val = metric_dict[i]
            t = ax.text(0.05, 0.88-0.12*ct, f"${i} = {val:.2f}$", 
                    fontsize = 12, color = 'k', transform = ax.transAxes)
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
            ct+=1
    return metric_dict
    
def lin_regression(x, y):
    ''' add linear regression line
        calculate R^2 (coefficient of determination)
    '''
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_sq = r_value**2
    y_fit = intercept + slope * x
    
    return r_sq, y_fit

def regplot_R2(varX, varY, df):
    """
    plot regression plot with R2 on it.
    
    Auguments: 
    varX: colname in df for x variable
    varY: colname in df for y variable
    """
    
    ax = sns.regplot(x = df[varX], y = df[varY])
    
    r_sq, y_fit = lin_regression(df[varX].values, df[varY].values)

    ax.text(0.05, 0.95, 'R\u00b2 = {0:.2f}'.format(r_sq), fontweight= 'bold',
         fontsize = 12, color = 'k', transform = ax.transAxes)
    
    return ax

def regplot_r(varX, varY, df):
    """
    plot regression plot with Pearson's r on it.
    
    Auguments: 
    varX: colname in df for x variable
    varY: colname in df for y variable
    """
    
    ax = sns.regplot(x = df[varX], y = df[varY])
    pr, _ = stats.pearsonr(df[varX].values, df[varY].values)

    ax.text(0.05, 0.95, 'r = {0:.2f}'.format(pr), fontweight= 'bold',
         fontsize = 12, color = 'k', transform = ax.transAxes)
    
    return ax
