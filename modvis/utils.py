'''
General Functions.
'''
import os, re, scipy
import numpy as np
import pandas as pd
import h5py
from math import cos, sin, asin, sqrt, radians
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, mannwhitneyu
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# from calendar import isleap
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
import dataretrieval.nwis as nwis
import sys
# sys.path.append("..")

# import objectivefunctions as ofs
import modvis.objectivefunctions as ofs

def insertNA(df, dt_col, freq=None):
    '''
    Automatically insert NAs for missing data.
    Arguments:
      - df. A dataframe contains the missing data.
      - dt_col. Column name in the df which contains time series.
    '''
    df[dt_col] = pd.to_datetime(df[dt_col])
    ## remove duplicates in the datetime index
#     df = df[~df[dt_col].duplicated()]
    df.drop_duplicates(subset=dt_col, inplace=True)
    df.sort_values(by = dt_col, inplace = True)
    # drop na
    # df = df.loc[df.index.notnull()]
    df.dropna(subset=[dt_col], inplace=True)
    ## create a new index
    start_time = df[dt_col].iloc[0]
    end_time = df[dt_col].iloc[-1]
    # ic(start_time)
    # ic(end_time)
    
    # find frequency
    if freq is None:
        ## get the highest frequency in sec
        freq_list = df[dt_col].diff().dt.seconds.value_counts()

        # the first freq with the most samples on the list may not be the highest freq
        if freq_list.index[0] == freq_list.index.min():
            freq = freq_list.index[0]
        else:
            freq = freq_list.index.min()
    logging.info(f'Found the highest frequency: {freq} sec')
    new_index = pd.date_range(start_time, end_time, freq=pd.DateOffset(seconds = freq),
                              name = dt_col)
    ## reindex and reset index
    df = df.set_index(dt_col).reindex(new_index).reset_index()

    return df

def days_between(d0,d1, noLeap=False):
    """calculates days between two dates
        Parameters:
            d0, datetime.date()
                starting date
            d1, datetime.date() object
                ending date
            noLeap, bool
                whether to consider leap years. 
                If True, only 365 days are considered each year.
                
        Returns:
            days in between dates.
    """
    delta = d1-d0
    if noLeap:
        # remove one day from leap year 
        start_year = d0.year
        end_year = d1.year
        leap_years = list(range(start_year + (4 - start_year % 4), end_year + 1, 4))
        
        return delta.days - len(leap_years)
    else:
        return delta.days

def load_nwis(service='dv', parameterCd='00060', SI_unit=True, insert_NA=True, rm_tz=True, **kwargs):
    """Load USGS streaflow from NWIS.
        Parameters:
            service, str
                NWIS service including dv, iv...
            parameterCd, str
                NWIS  parameter code including 00060, 00065, 00010 ...
            SI_unit, bool
                convert units to SI if true
            insert_NA, bool
                insert NAs where data is missing. Defaults to True.
            rm_tz, bool
                Remove time zone information. This may be necessary when comparing time series with vs. without time zones. 
                For sub-daily data, recommend to keep the timezone information.
            Other kwargs are the same in nwis.get_record().
            
        Returns:
            A dataframe that is cleaned and standardized. 
    """
    df = nwis.get_record(service=service, parameterCd=parameterCd, **kwargs)
    assert len(df) > 0, "Returned empty df!"
    
    # remove outliers and nan index
    df[df.iloc[:, 0] < 0 ] = np.nan
    df = df[df.index.notnull()]
    
    if SI_unit:
        if parameterCd == "00060" and '00060_Mean' in df.columns:
            # convert discharge from ft^3/s to m^3/d
            df['Discharge [m^3/d]'] = df['00060_Mean'] * 2446.58
            df = df[['Discharge [m^3/d]']]
    
    if rm_tz:
        df.reset_index(inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        df.set_index('datetime', inplace=True)
        
    if insert_NA and service=='dv':
        date_ind = pd.date_range(start = df.index[0], end = df.index[-1])
        new_df = pd.DataFrame(index = date_ind)
        df = df.join(new_df, how = 'outer')   
        
    return df


def mark_start_end(df, freq='1D'):
    """Mark the start and end of the dateframe time series. This can be used in combination with
    utils.gantt_plot().

    Parameters:
        df: pandas.DataFrame
            dataframe with datetime index and at least one column of data
        freq: str
            frequency of the data. Defaults to '1D' (daily).

    Returns:
        start_dates: list of datetime
        end_dates: list of datetime
    """

    # drop NaN data first
    df.dropna(inplace=True)
    # create a mask for missing data (assumes data is spaced evenly)
    mask = df.index.to_series().diff() > pd.Timedelta(freq)    

    # find the date before the mask (i.e. new starting date), will get new ending date
    end_idx = list(np.where(mask.values)[0] - 1)

    # find the start and end dates for each available time period
    start_dates = [df.index[0]] + df.index[mask].to_list()
    end_dates = df.index[end_idx].to_list() + [df.index[-1]]

    assert len(start_dates) == len(end_dates)

    return start_dates, end_dates

def K_perm_conversion(K=None, k=None):
    """convert between hydraulic conductivity (K) and permeability (k)
    
    Parameters:
        K: float
            hydraulic conductivity in m/d
        k: float
            permeability in m^2
    Returns
        K or k depending on the inputs
    """
    # constants assuming T=25 degC
    mu = 8.9e-4 # Pa*s
    rho = 997.0 # kg/m^3
    g = 9.8067 # m/s^2

    if K is not None and k is None:
        K_ms = K / 86400 # convert m/d to m/s
        k = K_ms * mu / (rho*g)
        return k

    if k is not None and K is None:
        K_ms = k * rho * g / mu
        K_md = K_ms * 86400
        return K_md


def feather_bounds(bounds, buffer = 0.01):
    """
    slightly enlarge the bound so that it covers the entire watershed.
    
    Parameters:
        bounds, array like
            List of bounds [xmin,ymin,xmax,ymax]
        buffer, float
            Buffer used to enlarge bounds. Use same unit as bounds.
    Returns:
        feather_bounds, list
            feathered bounds.
    """
    feather_bounds = list(bounds[:])
    feather_bounds[0] = feather_bounds[0] - buffer
    feather_bounds[1] = feather_bounds[1] - buffer
    feather_bounds[2] = feather_bounds[2] + buffer
    feather_bounds[3] = feather_bounds[3] + buffer
    logging.info(f"added {buffer} deg to get new bounds: {feather_bounds}")
    
    return feather_bounds


def latlon2km(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    See SOF: https://gis.stackexchange.com/a/61985/121955
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def proj_to_model(origin, angle, coord):
    """convert CRS to model coordinates"""
    x_coord = coord[:, 0]
    y_coord = coord[:, 1]
    
    new_x = (x_coord - origin[0])*np.cos(angle) + (y_coord - origin[1])*np.sin(angle)
    new_y = (y_coord - origin[1])*np.cos(angle) - (x_coord - origin[0])*np.sin(angle)
    
    new_coord = np.stack([new_x, new_y]).T
    
    return new_coord

def model_to_proj(origin, angle, coord):
    """convert model coordinates to CRS"""
    x_coord = coord[:, 0]
    y_coord = coord[:, 1]
    
    new_x = origin[0] + x_coord * np.cos(angle) - y_coord * np.sin(angle)
    new_y = origin[1] + x_coord * np.sin(angle) + y_coord * np.cos(angle)
    
    new_coord = np.stack([new_x, new_y]).T
    
    return new_coord

def outlier_removal(df, freq, threshold = 0.5, plot = False):
    """remove outlier from timeseries data.
    Parameters:
        df, DataFrame with datetime index
        freq, int or offset aliases. see pandas.DataFrame.rolling for more details.
            window size, e.g., '1D', 2. Increase this to smooth median curve.
        threshold, float
            the difference threshold between rolling median and raw data. Smaller number 
            will remove more outliers.
        plot, bool
            if plot the raw vs removed data.
    Returns:
        A new dataframe with outlier removed.
    """
    if not isinstance(df.index, pd.DatetimeIndex): 
        raise ValueError("Data Series must have datetime as index.")
    roll_median = df.iloc[:, 0].rolling(freq).median().fillna(method='bfill').fillna(method='ffill')
    df['rolling median'] = roll_median
    difference = np.abs(df.iloc[:, 0] - df['rolling median'].values)
    outlier_idx = difference > threshold

    logging.info("Found {} outliers".format(sum(outlier_idx)))

    if plot and sum(outlier_idx)>0:
#         fig, ax = plt.subplots(figsize=(6,5))
        df.iloc[:, 0].plot(style='o', mfc = 'none')
        df.iloc[:, 0][outlier_idx].plot(style='r*', alpha = 0.5)
        df['rolling median'].plot(style= 'k--')
        df.iloc[:, 0][~outlier_idx].plot(style = 'c')

        plt.legend(['raw data', 'outlier', 'rolling median', 'outlier removed']) 
    
    if sum(outlier_idx)>0:
        df_new = df[~outlier_idx]    
        return df_new
    else:
        return df

def nodata_vminmax(array, nodata):
    """get vmin,vmax by removing nodata"""
    masked_array = np.ma.masked_equal(array, nodata, copy=False)
    vmin,vmax = masked_array.min(), masked_array.max()
    return vmin, vmax

def natural_sort(l): 
    """naturally sort numbers in string. e.g. 0.txt, 1.txt, 2. txt, 11.txt"""
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def custom_legend(colors, labels, style = 'line', lw = 1, ax=None, **kwargs):
    """add custom lines as legend"""
    if style == 'line':
        custom_elements = [mlines.Line2D([0], [0], color=i, lw=lw) for i in colors]     
    elif style == 'patch':
        custom_elements = [mpatches.Patch(facecolor = i, edgecolor = i) for i in colors]

    if ax is None:
        return plt.legend(custom_elements, labels, frameon= False, **kwargs)
    else:
        return ax.legend(custom_elements, labels, frameon= False, **kwargs)

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    
    Author: https://stackoverflow.com/a/29677616/9319184
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def print_hdf5(name, obj):
    '''
    view the tree structure of hdf5 file
    
    example: input_h5.visititems(print_structure)
    '''
    print(name)
    
def h5_tree(h5_obj):
    """view tree structure of HDF5"""
#     h5_file = h5py.File(file, "r")
    h5_obj.visititems(print_hdf5)
#     h5_file.close()

def print_xml(file):
    """Print xml file in a clean way."""
    print(''.join(file))

def get_metrics(obs_t, obs, simu_t, simu, metrics = 'all', start_date = None,
                end_date = None, epsilon=0, dropzero=False, **kwargs):
    """
    Calculate metrics between simulated and observed data.
    Parameters:
        obs_t: list or array
            observed datetime series
        obs: list or array
            observed numeric series
        simu_t: list or array
            simulated datetime series
        simu: list or array
            simulated numeric sereis
        start_date: datetime object
            starting date for metric calculation
        end_date: datetime object
            ending date for metric calculation
        epsilon: float
            small number added to the data to avoid zero issue when using log transformation (e.g., logNSE). 
            Recommended value is 1/100 of the mean observed flow (see Santos et al., 2018 HESS for discussion).
        dropzero: bool
            if True, drop zero values in observed data. This is useful for dataframe with artifacts of zero values 
            (e.g., when using the restarted output from ATS).
        **kwargs: keyword arguments
            keyword arguments {return_all = True} for KGE components calculation
    Returns:
        dict of metrics and dataframe used.
    
    """
    VALID_METRICS = ['pearsonr', 'R^2', 'RMSE', 'rRMSE', 'NSE', 'logNSE',
                     'bias', 'pbias', 'KGE','npKGE', 'mKGE']
    
    df1 = pd.DataFrame((np.array(obs_t),obs))
    #, columns=['datetime', 'q_obs']
    df1 = df1.T
    df1.columns = ['datetime', 'obs']
    df1.dropna(inplace = True)

    df2 = pd.DataFrame((np.array(simu_t), simu))
    #, columns=['datetime', 'q_obs']
    df2 = df2.T
    df2.columns = ['datetime', 'simu']
    df2.dropna(inplace = True)
    
    df = df1.merge(df2, how='inner', on = 'datetime')

    if dropzero:
        # drop rows with any zero values
        df = df[(df != 0).all(1)]
    
    if start_date != None or end_date != None:
        df = df[df.datetime < end_date].copy()
        df = df[df.datetime > start_date]
        
    assert(df.shape[0] > 2)

    df['obs'] = pd.to_numeric(df['obs'])
    df['simu'] = pd.to_numeric(df['simu'])
    
    if metrics == 'all':
        metrics = VALID_METRICS
    
    metrics_dict = {}
    for i in metrics:
        if i == 'pearsonr':
            val = pearsonr(df['obs'].values, df['simu'].values)[0]
        elif i == 'R^2':
            val = (pearsonr(df['obs'].values, df['simu'].values)[0])**2
        elif i == 'RMSE':
            val = sqrt(mean_squared_error(df['obs'].values, df['simu'].values))
        elif i == 'rRMSE':
            val = ofs.rrmse(df['obs'].values, df['simu'].values)
        elif i == 'NSE':
            val = ofs.nashsutcliffe(df['obs'].values, df['simu'].values)
        elif i == 'logNSE':
            # default epsilon is 1/100 of the mean observed flow
            if epsilon == 0:
                epsilon = df['obs'].mean() / 100
            val = ofs.lognashsutcliffe(df['obs'].values, df['simu'].values, epsilon=epsilon)
        elif i == 'bias':
            val = ofs.bias(df['obs'].values, df['simu'].values)
        elif i == 'pbias':
            val = ofs.pbias(df['obs'].values, df['simu'].values)
        elif i == 'KGE':
            val = ofs.kge(df['obs'].values, df['simu'].values, **kwargs)
        elif i == 'npKGE':
            val = ofs.kge_non_parametric(df['obs'].values, df['simu'].values, **kwargs)
        elif i == 'mKGE':
            val = ofs.mkge(df['obs'].values, df['simu'].values, **kwargs)
        else:
            raise KeyError(f"{i} not found in {VALID_METRICS}!")
        metrics_dict[i] = val
        
    return metrics_dict, df

def convertUSGSgauge(fname, var, datetime_col="datetime", discharge_col='Flow', rmnan = True, insertNa =True, drop_tz = True):
    """convert USGS discharge/ gauge height to SI unit"""
    df = pd.read_csv(fname)
    df.set_index(datetime_col, drop=False)
    
    if var == 'discharge':
        sub_df = df[[datetime_col, discharge_col]].copy()
        sub_df['Discharge [m^3/d]'] = sub_df[discharge_col]*2446.58 # convert from ft3/s to m3/d
        sub_df[datetime_col] = pd.to_datetime(sub_df[datetime_col])
        if drop_tz:
            # remove the UTC time zone
            sub_df[datetime_col] = sub_df[datetime_col].dt.tz_localize(None)
        sub_df.set_index(datetime_col, inplace = True, drop = False)
        
    if rmnan:
        sub_df = sub_df[sub_df.index.notnull()]        
        
    if insertNa:
        logging.info("insert nan")
        date_ind = pd.date_range(start = sub_df[datetime_col][0], end = sub_df[datetime_col][-1])
        new_df = pd.DataFrame(index = date_ind)
        sub_df = sub_df.join(new_df, how = 'outer')
        
    return sub_df

def forward_selected(data, response):
    """Linear model designed by forward selection. 
    Rank the variables with R^2 (or variance explanied). 
    Adding variable one by one until all variables are used.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared.
    The one explained the most variance comes out on top.
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        print('|variable|total R^2|increased R^2|')
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            print(best_candidate, '%.2f'%best_new_score, '%.3f'% (best_new_score-current_score))
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

def array2asc(file_array, file_asc, nx=600, ny=600, dx=100, model_origin = [0,0]):
    """
    Convert 2D numpy array to ASCII file that can be used by QGIS/ARCGIS.
    
    Arguments:
        - file_array: path to array file (read)
        - file_asc: path to asc file (write)
    """
    
    TheFile=open(file_asc,"w")
    TheFile.write("ncols {}\n".format(nx))
    TheFile.write("nrows {}\n".format(ny))
    TheFile.write("xllcorner {}\n".format(model_origin[0]))
    TheFile.write("yllcorner {}\n".format(model_origin[1])) 
    TheFile.write("cellsize {}\n".format(dx)) 
    TheFile.write("NODATA_value  0\n")

    TheFormat="{0} "

    ncols= nx
    nrows= ny

    table = []
    data =[]
    with open(file_array) as my_file:
        for line in my_file: #read line by line

             numbers_str = line.split() #split string by " "(space)
            #convert string to floats
             numbers_str_new = ["{0:.2f}".format(float(x)) for x in numbers_str]  #map(float,numbers_str) works too (convert feet to meter with factor of 0.3048)

             table.append(numbers_str_new) #store each string line

    for item in table[::-1]:
        #loop over each line
        for ele in item: #loop over each element in line
    #        print(ele) 
            data.append(ele) #store each element one by one 

    ## read into file
    for i in range(0, len(data), ncols): #loop over data with stepsize of ncols
    #    print(data[i:i+ncols])
        TheFile.write(" ".join(data[i:i + ncols])) #join element in list with space, and write into file
        TheFile.write("\n")#write new line

    TheFile.close()
    
    print('Exported ASCII file is in {}'.format(file_asc))
    
    return None

def mann_whitney_test(data1, data2, alpha=0.05):
    """
    perform a Mann-Whitney test, and return statistics and p-value
    Parameters:
        x: data1, dataframe timeseries
        y: data2, dataframe timeseries
        alpha: significance factor, default 0.05
    Returns:
        stat, p
    """
    stat, p = mannwhitneyu(data1, data2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    return stat, p
        
    return stat, p
