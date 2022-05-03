"""Functions for post-processing ATS models."""
import os, re, scipy, glob, itertools
import numpy as np
import pandas as pd
import h5py
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timedelta
import time
from calendar import isleap
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
import sys
# sys.path.append(os.environ['MYFUNC_DIR'])
# sys.path.append("../myfunctions")
# sys.path.append(".")
# import utils as utils
# import colors as fcolors
# import ats_xdmf as xdmf
from myfunctions import utils
import myfunctions.colors as fcolors
import myfunctions.ats_xdmf as xdmf

colors = fcolors.colors('matplotlib')
rho_m = 55500 # moles/m^3, water molar density
# rho_m = 55000 # moles/m^3, water molar density. Make sure this is consistent with what's in the xml file!!
rho = 997
g = 9.80665

def get_snow_cover(model_dir, **kwargs):
    """Calculate snow cover data from model output. Return a dataframe with snow cover percentage."""
    surface_vis = xdmf.VisFile(model_dir, domain="surface", filename= model_dir + "ats_vis_surface_data.h5", 
                           mesh_filename= "ats_vis_surface_mesh.h5", 
                           model_time_unit='d')
    
    snow_frac = surface_vis.getArray("surface-area_fractions.cell.1")
    surface_vis.loadMesh()
    surface_area = surface_vis.volume
    snow_cover_area = np.matmul(snow_frac, surface_area) 
    snow_cover_pct = snow_cover_area/np.sum(surface_area)
    
    times = surface_vis.times
    t = rmLeapDays(times, **kwargs)
    
    df = pd.DataFrame({'datetime': t, 'snow_cover_pct [-]':snow_cover_pct})
    df.set_index('datetime', inplace=True, drop = True) 
    
    return df

def plot_precip(df, add_on = None, add_on_label=None, add_on_color=None, itime = None, ax = None, left_ylim = None):
    """plot rain, snowmelt and ET on the same plot with option of adding time bar."""
    fontsize = 14
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = (8,4))
#     xlim = [datetime(2015,10,1), datetime(2016,9,30)]

#     ax1 = ax.twinx()
    ax.bar(df.index, df["rain precipitation [mm d^-1]"], lw=1)
    ax.bar(df.index, df["snowmelt [mm d^-1]"], lw=1, bottom = df["rain precipitation [mm d^-1]"])
    
    if add_on is not None:
        ax1 = ax.twinx()
        if add_on_color is None:
            add_on_color = colors[2]
        ax1.plot(df.index, df[add_on], color = add_on_color, lw=1)
        ax1.set_ylabel(add_on, fontsize = fontsize)
#         ax1.tick_params(axis='both', which='major', labelsize=12)   
        
    if itime is not None:
        if type(itime) is str:
            itime = datetime.strptime(itime, "%Y-%m-%d")
        ax.axvline(x = itime, color = 'crimson', zorder = 99)
        ax.set_title(itime.date(), pad = 20, fontsize = 16) 
    if left_ylim is not None:
        ax.set_ylim(left_ylim)
    else:
        ax.set_ylim([0,70])
#     ax1.set_ylim([0, 5])
    ax.set_xlim([df.index[0], df.index[-1]])
    ax.invert_yaxis()
    ax.set_ylabel('Rain/snowmelt ($mm/d$)', fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=12)

    if add_on is None:
        labels = ["Rain", "Snowmelt"]
        line_colors = colors[:2]
    else:
        if add_on_label is None:
            labels = ["Rain", "Snowmelt", add_on]
        else:
            labels = ["Rain", "Snowmelt", add_on_label]
        line_colors = colors[:2] + [add_on_color]
    
    utils.custom_legend(line_colors, labels = labels,loc = 'upper left', 
                        bbox_to_anchor =(0.01 , 1.12), ncol =
                        len(labels), ax = ax)
    ax.tick_params(axis=u'x', which=u'minor',length=0)
    ax.set_xlim(df.index[0],df.index[-1])
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
    
    if add_on is None:
        return ax
    else:
        return ax, ax1


def xyz_from_cellID(ordered_centroids, mapping, cellID):
    """get x,y,z coordinates based on cell id"""
    col_ids, layer_ids = colID_from_cellID(mapping, cellID)
    xyz = []
    for icol,ilayer in zip(col_ids, layer_ids):
        ixyz = ordered_centroids[icol, ilayer, :]
        xyz.append(ixyz)
    return xyz
    
def colID_from_cellID(mapping, cellID):
    """get column, layer id based on cell id
    Parameters:
        mapping, array
            object returned from xdmf.structuredOrdering()
        cellID, list or array like
            cell id
    Returns:
        column id and layer id
    """
    col_ids = []
    layer_ids = []
    for i in cellID:
        icol, ilayer = np.where(mapping == i)
        col_ids.append(icol[0])
        layer_ids.append(ilayer[0])
        
    if len(col_ids) == 0 or len(layer_ids) == 0:
        raise RuntimeError("cell id not found in mesh!")
    else:
        return col_ids, layer_ids 


def combine_hdf5(dir1, dir2, vis_file = "ats_vis_data.h5"):
    """combine two hdf5 files into one."""
    
    start_time = time.time()
    sub_const_list = ['base_porosity.cell.0', 'capillary_pressure_gas_liq.cell.0', 'depth.cell.0', 
           'mass_density_liquid.cell.0', 'molar_density_liquid.cell.0', 'permeability.cell.0', 
             'plant_wilting_factor.cell.0', 'porosity.cell.0', 'relative_permeability.cell.0', 
              'rooting_depth_fraction.cell.0', 'saturation_gas.cell.0', 'viscosity_liquid.cell.0']
    surface_const_list = ['surface-aspect.cell.0', 'surface-manning_coefficient.cell.0', 'surface-mass_density_liquid.cell.0', 
           'surface-molar_density_liquid.cell.0', 'surface-overland_conductivity.cell.0', 'surface-relative_permeability.cell.0', 
             'surface-slope_magnitude.cell.0', 'surface-source_molar_density.cell.0']
    const_list = sub_const_list + surface_const_list

    fname_out = vis_file.replace('.h5', '-combo.h5')
    
    output_h5 = h5py.File(dir1 + fname_out, "w")
    logging.info(f"Generating combined file: {dir1 + fname_out}")
    
    # read the first file
    logging.info(f"copying the first file: {dir1 + vis_file}")
    input_h5 = h5py.File(dir1 + vis_file, "r")
    groups = list(input_h5.keys())
    groups = [e for e in groups if e not in const_list]

    times = list(input_h5[groups[0]].keys())
    times = sorted(times, key = lambda time: float(time))
    
    for i_group in groups[:]:
#         print(i_group)
        group_id = output_h5.require_group(i_group)
        datasets = list(input_h5[i_group].keys())
        for i_dataset in datasets:
            input_h5.copy("/" + i_group + "/" + i_dataset,
                          group_id, name=i_dataset)
    input_h5.close()
    
    # read the second file
    logging.info(f"copying the second file: {dir2 + vis_file}")
    input_h5 = h5py.File(dir2 + vis_file, "r")
    
    new_times = list(input_h5[groups[0]].keys())
    new_times = sorted(new_times, key = lambda time: float(time))

    comm_time = list(set(times).intersection(new_times))
    if len(comm_time) > 0:
        new_times = [e for e in new_times if e not in comm_time]   
    
    for i_group in groups:
#         print(i_group)
        group_id = output_h5.require_group(i_group)
    #     datasets = list(input_h5[i_group].keys())

        for i_dataset in new_times:
            input_h5.copy("/" + i_group + "/" + i_dataset,
                          group_id, name=i_dataset)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
   
    logging.info(f"Done! Time spent: {elapsed_time:.2f} s")
    input_h5.close()
    output_h5.close()

def get_major_nlcd(df, key = 'coverage [%]', thresh=90):
    """get major land cover species"""
    df = df.sort_values(by = key, ascending=False)
    i = 0
    while i < df.shape[0]:
        sum_major = df[key].values[:i].sum()
        if sum_major > thresh:
            return df.index[:i].values, df.index[i:].values
            break        
        i +=1

def combine_obs_output(old_dir, new_dir):
    """combine model observations after restart"""
    df_old = load_waterBalance(old_dir, cumsum = False)
    df_new = load_waterBalance(new_dir, cumsum = False)
    
    df_old_sub = df_old.loc[:, 'time [d]':'(Pr+SM-ET-Q) [m/d]']
    df_new_sub = df_new.loc[:, 'time [d]':'(Pr+SM-ET-Q) [m/d]']
    
    assert(df_old_sub.shape[-1] == df_new_sub.shape[-1])
    df = pd.concat([df_old_sub, df_new_sub]) 
    
    return df
    

def gen_column_data_h5(z0, z1, h5file, unsat_thickness=0):    
    """create the column_data file to initialized the presure field in the subsurface."""
    z = np.array([0, 100])
    pres = 101325. + (np.array([0.,100.]) - unsat_thickness)*rho*g

    with h5py.File(h5file, 'a') as fid:
        fid.create_dataset('z', data=z)
        fid.create_dataset('pressure', data=pres)    

def plot_timestep(work_dir, fname_run_log=None):
    """read run log generated by ATS and plot timestep over time"""
    if not os.path.exists(work_dir):
        raise NameError(f"{work_dir} could not be found!")
   
   # look for log files following the sequence slurm*.out, run*.log ...
    if fname_run_log is None:
        log_files = glob.glob(work_dir + "job*.out")
        if len(log_files) == 0:
            log_files = glob.glob(work_dir + "slurm*.out")
        if len(log_files) == 0:
            log_files = glob.glob(work_dir + "run*.log")
        if len(log_files) == 0:
            log_files = glob.glob(work_dir + "out*.log")
        if len(log_files) == 0:
            raise RuntimeError(f"Could not find log file in dir: {work_dir}!")
        log_files.sort(key = os.path.getmtime)
        log_file_path = log_files[-1]    
    elif type(fname_run_log) is str:
        log_file_path = work_dir + fname_run_log
    # print(log_file_path)
    logging.info(f"found file {log_file_path}!")

    with open(log_file_path, "r") as f:
        searchlines = f.readlines()
    assert len(searchlines) > 0, "log file is empty!"

    time_dt = []
    step_dt=[]
    dt=[]

    for line in searchlines[:]:
        if "Cycle" in line and "Time" in line: 
            splitline = re.split("\s+|\x1b|,", line)
            # print(splitline)
            cycle_idx = splitline.index("Cycle")
            time_idx = splitline.index("Time")
            dt_idx = splitline.index("dt")
            istep = splitline[cycle_idx + 2]        
            itime = splitline[time_idx + 3]        
            idt = splitline[dt_idx + 3]
            dt.append(idt)
            step_dt.append(istep)
            time_dt.append(itime)  
            
        if "TimeMonitor" in line and "|" in line:
            splitline = re.split("\s+|,", line)
            ncores = splitline[-3]
            
        if "Simulation Driver  |  cycle" in line:
            splitline = re.split("\s+|,", line) 
            walltime = float(splitline[4])/3600 #convert s to h

    dt = np.array([float(i) for i in dt])
    assert len(dt) > 0, "0 timestep found!"
    time_dt = np.array([float(i) for i in time_dt])
    nyear = np.ptp(time_dt)/365 # cnvt d to year
    
    step_dt = [float(i) for i in step_dt] 
    df = pd.DataFrame(np.stack([step_dt, dt, time_dt]).T, columns = ['cycle', 'timestep [d]', 'times [d]'])
    ave_dt = np.array(dt).mean()*24 # convert from day to hour
    
    # plot
    fig,ax = plt.subplots(1,1, figsize=(8,6))

    ax.plot(step_dt, dt, 'k', lw = 0.5)
    ax.set_ylabel('Timestep (day)')
    ax.set_xlabel('Cycle')
    ax.set_yscale('log')
    
    try:
        walltime_per_year = walltime / nyear
        ax.set_title(f"Cores: {ncores}; wallclock time [h]: {walltime:.2f}; ave ts [h]: {ave_dt:.2f}; ave walltime/yr [h]: {walltime_per_year:.2f}")
    except:
        pass

    ax1 = ax.twinx()
    ax1.plot(step_dt, time_dt, 'r', lw=0.5)
    ax1.set_ylabel('Cumulative time (day)', color = 'red')
    ax1.tick_params('y', colors='r')   
    
    return fig, ax, ax1, df

def isleap_and_31Dec(t):
    """find Dec. 31 in leap year"""
    return [isleap(i.year) & (i.month == 12) & (i.day == 31) for i in t]

def rmLeapDays(time_ats, origin_date = "1980-01-01", noleap = True, freq = "D"):
    """remove Dec31 from leap years so that each year = 365 d. This applies to ATS and PRMS
    Parameters:
        time_ats: list
            list of days or hours
        origin_date: datetime str in "%Y-%m-%d", default is 1980-1-1
            original datetime. In ATS, this depends on the starting date of Daymet. 
    Returns:
        datetime array with leap days removed.
    """
    # round to the nearest int, this maybe necessary for
    # floating point erros
    time_ats = np.rint(time_ats)

    if type(origin_date) is str:
        origin_date = datetime.strptime(origin_date, "%Y-%m-%d")
    elif isinstance(origin_date, dt.datetime):
        pass
    else:
        raise ValueError("Must be either datetime.datetime object or %Y-%m-%d string!")
    if freq == 'D':
        model_start_year = time_ats[0]//365
        model_start_doy = time_ats[0]%365
    elif freq == 'H':
        model_start_year = time_ats[0]/24 //365
        model_start_doy = time_ats[0]/24 %365   
    else:
        raise RuntimeError("Must be either D or H freq!")
    model_start_date = origin_date.replace(year = origin_date.year + int(model_start_year))
    model_start_date = model_start_date + timedelta(days = int(model_start_doy))
    
    if freq == 'D':
        datetime_ats = [model_start_date + timedelta(days = int(i)) for i in time_ats - time_ats[0]]
    elif freq == 'H':
        datetime_ats = [model_start_date + timedelta(hours = int(i)) for i in time_ats - time_ats[0]]    
    # shift date so that 12/31 is removed from leap years
    if noleap:
        datetime_ats = np.array(datetime_ats)
        mask = isleap_and_31Dec(datetime_ats)
        while True in mask:
            ind = mask.index(True) # find the first index of 12-31 day
            datetime_ats[ind:] = datetime_ats[ind:] + timedelta(days = 1)  
            mask = isleap_and_31Dec(datetime_ats)   
            
    return datetime_ats

def sec_noleap(origin_date = "1980-01-01", end_year = 2020, freq = "D"):
    """get seconds since original date without leap day (i.e. 1y = 365d)"""
    if type(origin_date) is str:
        origin_date = datetime.strptime(origin_date, "%Y-%m-%d")
    elif isinstance(origin_date, pd.datetime):
        pass
    else:
        raise ValueError("Must be either datetime.datetime object or %Y-%m-%d string!")
    nyear = end_year - origin_date.year + 1
    days = np.arange(0, nyear*365, 1)
    hours = np.arange(0, nyear*365*24, 1)
    if freq == "D":
        seconds = days*86400
    elif freq == "H":
        seconds = hours*3600
    else:
        raise ValueError("Must be 'H' or 'D' frequency!")

    datetime_noleap = rmLeapDays(days, origin_date)
    
    if freq == 'H':
        datetime_noleap_sec = [pd.date_range(datetime_noleap[i], datetime_noleap[i] + timedelta(days = 1), freq = freq)[:-1].tolist() for i in range(len(datetime_noleap))]
        datetime_noleap = np.array(list(itertools.chain.from_iterable(datetime_noleap_sec)))
        
    ats_datetime = pd.DataFrame(np.vstack([datetime_noleap, seconds]).T, columns=['datetime', 'time [s]'] )
    ats_datetime.set_index('datetime', inplace = True)    
    ats_datetime['time [s]'] = pd.to_numeric(ats_datetime['time [s]'])
    
    return ats_datetime

def get_subbasin_value(vis_data, varname, subbasin_cells, volume, times, weighted = True, out_file = None, plot = False, ax = None):
    """map values to subbasin. For comparison with SWAT
    Parameters:
        vis_data, surface vis from ats_xdmf.VisFile()
        varname, str
            variable name in the surface vis
        subbasin_cells, array
            cells dict for each subbasin
        volume, float
            cell areas
        times, datetime
        weighted, bool
            whether to take weighted average
    Returns:
        DataFrame with each subbasin values.
    """
    
    data = vis_data.getArray(varname)
    if 'transpiration' in varname or 'precip' in varname or 'snow-melt' in varname:
        #convert m/s to mm/d
        data = data*86400*1000 
        unit = '[mm/d]'
    elif 'flux' in varname:
        #convert mol/s to m^3/d
        data = data / rho_m *86400 
        data = data/ volume *1000 # m3/d to mm/d
        unit = '[mm/d]'
    elif "snow-water_equivalent" in varname:
        # convert m to mm
        data = data*1000 
        unit = '[mm]'
    else:
        logging.info("No unit convertion.")
        unit = None
    
    keys = list(subbasin_cells.keys())
    subbasin_value = {}
    for key in keys:
        sub_cell = subbasin_cells[key]
        sub_volume = volume[sub_cell]
        if weighted:
            # perform weighted average
            ivalue = np.dot(data[:, sub_cell],sub_volume)/sub_volume.sum()
        else:
            ivalue = data[:, sub_cell].mean(axis = 1)
        subbasin_value['subbasin_' + str(key)] = ivalue
        
    subbasin_df = pd.DataFrame(subbasin_value)
    icol = varname if unit == None else varname+ ' ' + unit
    subbasin_df[icol] = np.dot(data[:, :], volume)/volume.sum()
#     subbasin_df['datetime'] = rmLeapDays(time)
    subbasin_df['datetime'] = times
    subbasin_df.set_index('datetime', inplace = True)
    
    if out_file != None:
#         out_file = "subbasin_" + varname + "_.csv"
        subbasin_df.to_csv(out_file)
    
    if plot:
        if ax == None:
            fig, ax = plt.subplots(1,1, figsize=(6, 4))
        subbasin_df.iloc[:, :-1].plot(lw = 0.5, alpha = 0.5, ax = ax)
        subbasin_df[icol].plot(lw = 0.5, color = 'k', ax = ax)
    
    if ax == None:
        return subbasin_df, fig, ax
    else:
        return subbasin_df


def surfaceArea(model_dir):
    """get surface area of ATS model"""
    h5_files = glob.glob(model_dir + "*surface_data.h5")
    if len(h5_files) == 1:
        with h5py.File(h5_files[0]) as f:
            a_key = list(f['surface-cell_volume.cell.0'].keys())[0] # pick any timestamp
            surface_area = f['surface-cell_volume.cell.0'][a_key][:].sum() # m^2  
    else:
        raise KeyError("*surface_data.h5 could not be found!")
    return surface_area
    
def load_waterBalance(model_dir, WB_filename = "water_balance.dat", timestep =
                      'D', UTC_time = None, resample_freq = None, canopy =
                      False, origin_date = "1980-01-01", noleap = True, cumsum = True,
                      restart_dir = None, out_file = None, plot = False,
                      subcatchment=False, catchment_area=None,
                      **kwargs):
    """read ATS output files, new dataframe format
    Parameters:
        model_dir: str
            ATS model directory
        WB_filename: str
            water balance filename
        timestep: str, ['D', 'h']
            frequency of output. Default to daily.
        UTC_time: int, 
            hours needed to shift time series index to match UTC time zone
        resample_freq: str,
            freq used to resample time series. Options include 'D'
        canopy: bool,
            whether to include canopy module
        origin_date: datetime, default is 1980-1-1
            Original datetime in Daymet forcing time column
        noleap: bool, default is True
            Assume model ignores the leap year to be consistent with Daymet product
        cumsum: bool,
            If true, generate cumsum for vairables
        restart_dir: list or None
            If list, merge with restarted model output
        out_file: str or None
            If string, save water balance in a csv file.
        plot: bool,
            If true, plot water balance plots
    Returns:
        datetime, data, and dataframe(datetime, data)
    
    """
    if subcatchment:
        surface_area = catchment_area
    else:
        surface_area = surfaceArea(model_dir)
    assert(surface_area is not None)

    df = load_output(model_dir, WB_filename, timestep, origin_date, **kwargs)
    
    if restart_dir is not None:
        assert(type(restart_dir) is list)
        for idir in restart_dir:
            df_rst = load_output(idir, WB_filename, timestep, origin_date,
                                 **kwargs)
            if df.shape[-1] != df_rst.shape[-1]:
                logging.warning("waterbalance.out file columns does not match! Trying merging anyway.")
            df = pd.concat([df, df_rst], join = "inner")    
            
        # remove duplicates in time index        
        df = df[~df.index.duplicated(keep='first')]      
   
    # get names consistent
    if "watershed boundary discharge [mol d^-1]" not in df.columns:
        try:
            df["watershed boundary discharge [mol d^-1]"] = df["net runoff [mol d^-1]"]
        except:
            logging.info("counld not find the name 'watershed boundary discharge' in the output!")

    if "river discharge [mol d^-1]" in df.columns:
        df["river discharge [m^3/d]"] = df['river discharge [mol d^-1]'] / rho_m
        df["river discharge [m/d]"] = df['river discharge [m^3/d]'] / surface_area

    if "net groundwater flux [mol d^-1]" in df.columns:
        df["net groundwater flux [m^3/d]"] = df['net groundwater flux [mol d^-1]'] / rho_m
        df["net groundwater flux [m/d]"] = df['net groundwater flux [m^3/d]'] / surface_area

    # convert units
    if timestep == 'H':
        df['watershed boundary discharge [mol d^-1]'] = df['watershed boundary discharge [mol h^-1]']*24        
        df['total evapotranspiration [m d^-1]'] = df['total evapotranspiration [m h^-1]']*24        
        df['rain precipitation [m d^-1]'] = df['rain precipitation [m h^-1]']*24
        df['snow precipitation [m d^-1]'] = df['snow precipitation [m h^-1]']*24
        df['snowmelt [m d^-1]'] = df['snowmelt [m h^-1]']*24
        if canopy:
            df['snow evaporation [m d^-1]'] = df['snow evaporation [m h^-1]']*24  
            df['surface evaporation [m d^-1]'] = df['surface evaporation [m h^-1]']*24 
            df['transpiration [m d^-1]'] = df['transpiration [m h^-1]']*24 
            df['canopy evaporation [m d^-1]'] = df['canopy evaporation [m h^-1]']*24
            df['canopy drainage [m d^-1]'] = df['canopy drainage [m h^-1]']*24
            df['water to surface [m d^-1]'] = df['water to surface [m h^-1]']*24
            df['snow to surface [m SWE d^-1]'] = df['snow to surface [m SWE h^-1]']*24
        
    df['rain precipitation [mm d^-1]'] = df['rain precipitation [m d^-1]']*1000
    df['snow precipitation [mm d^-1]'] = df['snow precipitation [m d^-1]']*1000
    df['snowmelt [mm d^-1]'] = df['snowmelt [m d^-1]']*1000
    df['watershed boundary discharge [m^3/d]'] = df['watershed boundary discharge [mol d^-1]'] / rho_m
    df['watershed boundary discharge [m/d]'] = df['watershed boundary discharge [m^3/d]'] / surface_area
    df["total evapotranspiration [mm d^-1]"] = df["total evapotranspiration [m d^-1]"] *1000         
    df['snow water content [m]'] = df['snow water content [mol]'] / rho_m / surface_area
    df['surface water content [m]'] = df['surface water content [mol]'] / rho_m / surface_area
    df['subsurface water content [m]'] = df['subsurface water content [mol]'] / rho_m / surface_area        
    if not canopy:            
        df['total water content [m]'] = df['surface water content [m]'] + df['subsurface water content [m]'] + df['snow water content [m]']
    else:
        df["canopy water content [m]"] = df["canopy water content [mol]"]/ rho_m / surface_area
        df['(Ps_thru-SM-Es) [m/d]'] = df['snow to surface [m SWE d^-1]'] - df['snowmelt [m d^-1]'] -df['snow evaporation [m d^-1]']
        df['total water content [m]'] = df['surface water content [m]'] + df['subsurface water content [m]'] + df['snow water content [m]'] + df["canopy water content [m]"]
      
    if "net groundwater flux [mol d^-1]" in df.columns:
        df['(Pr+SM-ET-Q-gw) [m/d]'] = df['rain precipitation [m d^-1]'] + df['snowmelt [m d^-1]'] -\
                            df['total evapotranspiration [m d^-1]'] - df['watershed boundary discharge [m/d]'] -\
                            df['net groundwater flux [m/d]']
        df['(Pr+Ps-ET-Q-gw) [m/d]'] = df['rain precipitation [m d^-1]'] + df['snow precipitation [m d^-1]'] -\
                            df['total evapotranspiration [m d^-1]'] - df['watershed boundary discharge [m/d]'] -\
                            df['net groundwater flux [m/d]']
    # calculate water balance
    df['(Pr+SM-ET-Q) [m/d]'] = df['rain precipitation [m d^-1]'] + df['snowmelt [m d^-1]'] -\
                        df['total evapotranspiration [m d^-1]'] - df['watershed boundary discharge [m/d]']
    df['(Pr+Ps-ET-Q) [m/d]'] = df['rain precipitation [m d^-1]'] + df['snow precipitation [m d^-1]'] -\
                        df['total evapotranspiration [m d^-1]'] - df['watershed boundary discharge [m/d]']
    df['(Ps-SM) [m/d]'] = df['snow precipitation [m d^-1]'] - df['snowmelt [m d^-1]']

    if resample_freq is not None:
        variables = ['rain precipitation [m d^-1]','snow precipitation [m d^-1]','snowmelt [m d^-1]', 'watershed boundary discharge [m/d]', 'watershed boundary discharge [m^3/d]', 'total evapotranspiration [m d^-1]', 'max ponded depth [m]', 'SWE [m]']
        df_rs = df.loc[:, variables].copy()
        logging.info(f"Resample to {resample_freq} frequency")
        df_rs = df_rs.resample(resample_freq).mean()
        df_rs.dropna(inplace = True)
        
        if UTC_time is not None:
            df_rs.index = df_rs.index.shift(UTC_time, freq = 'H')     
    
    if cumsum:
        
        if "net groundwater flux [mol d^-1]" in df.columns:
            df["cum_groundwater flux [m]"] = df['net groundwater flux [m/d]'].cumsum()
            df['cum_(Pr+Ps-ET-Q-gw) [m]'] = df['(Pr+Ps-ET-Q-gw) [m/d]'].cumsum()
        if canopy:
            df['cum canopy mass change [m]'] = df["canopy water content [m]"].cumsum()
            df['cum_snow evaporation [m]'] = df['snow evaporation [m d^-1]'].cumsum()
            df["cum_snow to surface [m]"] = df['snow to surface [m SWE d^-1]'].cumsum()
            df['cum_(Ps_thru-SM-Es) [m]'] = df['(Ps_thru-SM-Es) [m/d]'].cumsum()
            df['cum_(P-Pt+S-St-Ec-drainage) [m]'] = (df['rain precipitation [m d^-1]'] - df["water to surface [m d^-1]"] + df["snow precipitation [m d^-1]"] - df["snow to surface [m SWE d^-1]"] - df["canopy evaporation [m d^-1]"] - df["canopy drainage [m d^-1]"]).cumsum()
        df["cum_rain precipitation [m]"] = df['rain precipitation [m d^-1]'].cumsum()
        df["cum_snow precipitation [m]"] = df['snow precipitation [m d^-1]'].cumsum()
        df["cum_snowmelt [m]"] = df['snowmelt [m d^-1]'].cumsum()
        df["cum_ET [m]"] = df["total evapotranspiration [m d^-1]"].cumsum()
        df['cum_overland flux [m]'] = df['watershed boundary discharge [m/d]'].cumsum()
        df['cum_(Pr+SM-ET-Q) [m]'] = df['(Pr+SM-ET-Q) [m/d]'].cumsum()
        df['cum_(Pr+Ps-ET-Q) [m]'] = df['(Pr+Ps-ET-Q) [m/d]'].cumsum()
        df['cum_(Ps-SM) [m]'] = df["(Ps-SM) [m/d]"].cumsum()
        df['cum snow mass change [m]'] = df['snow water content [m]'] - df['snow water content [m]'].values[0]
        df['cum water mass change [m]'] = df['total water content [m]'] - df['total water content [m]'].values[0]

        if "net groundwater flux [mol d^-1]" in df.columns:
            df['water mass error [m]'] = df['cum_(Pr+Ps-ET-Q-gw) [m]'] - df['cum water mass change [m]']
        else:
            df['water mass error [m]'] = df['cum_(Pr+Ps-ET-Q) [m]'] - df['cum water mass change [m]']

        if not canopy:
            df['snow mass error [m]'] = df['cum_(Ps-SM) [m]'] - df['cum snow mass change [m]']
        else:
            df['snow mass error [m]'] = df['cum_(Ps_thru-SM-Es) [m]'] - df['cum snow mass change [m]']
            
    if plot:
        
        fig, axes = plt.subplots(3, 1, figsize=(8,8))

        ax = axes[0]
        df.plot(y=["rain precipitation [m d^-1]", "snowmelt [m d^-1]" ], color
                = colors[:2],
        ax = ax)
        ax1 = ax.twinx()
        if "river discharge [mol d^-1]" in df.columns:
            df.plot(y=["total evapotranspiration [m d^-1]", 
                   "river discharge [m/d]"], color =
                colors[2:4], ax = ax1)
        else:
            df.plot(y=["total evapotranspiration [m d^-1]", 
                   "watershed boundary discharge [m/d]"], color =
                colors[2:4], ax = ax1)
        ax.set_ylabel('Incoming flux [m/d]')
        ax1.set_ylabel('Outgoing flux [m/d]')
        ax.legend(loc='upper left', fontsize = 10, bbox_to_anchor = (0.0,1.3), frameon = False)
        ax1.legend(loc='upper left', fontsize = 10, bbox_to_anchor = (0.5,1.3),
                  frameon = False)
        ax.xaxis.set_ticklabels([])
        ax.invert_yaxis()
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_xlabel('')

        ax = axes[1]
        if "net groundwater flux [mol d^-1]" in df.columns:
            df.plot(y=["cum_rain precipitation [m]", "cum_snow precipitation [m]", "cum_ET [m]", 
                   "cum_overland flux [m]", "cum_groundwater flux [m]", 'cum_(Pr+Ps-ET-Q-gw) [m]'], ax = ax)
        else:
            df.plot(y=["cum_rain precipitation [m]", "cum_snow precipitation [m]", "cum_ET [m]", 
                   "cum_overland flux [m]", 'cum_(Pr+Ps-ET-Q) [m]'], ax = ax)
        df.plot(y = ['cum water mass change [m]', 'water mass error [m]'], style = ['-.','-'], color = ['k', 'grey'], ax = ax)
        ax.set_ylabel('CumFlux [m]')
        ax.legend(loc='upper left', fontsize = 10, bbox_to_anchor = (1.0,1.15), frameon = False)
        max_error = max(abs(df['water mass error [m]'].max()), abs(df['water mass error [m]'].min()))
#         print(max_error)
        ax.set_title(f'max error = {max_error*1000:.2f} mm')
        ax.xaxis.set_ticklabels([])
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_xlabel('')

        ax = axes[2]
        if not canopy:
            df.plot(y=["cum_snow precipitation [m]", "cum_snowmelt [m]", 'cum_(Ps-SM) [m]'], ax = ax)
            df.plot(y=['cum snow mass change [m]', 'snow mass error [m]'], style = ['-.','-'], color = ['k', 'grey'], ax = ax)
        else:
            df.plot(y=[ "cum_snow to surface [m]", "cum_snowmelt [m]", 'cum_snow evaporation [m]', 'cum_(Ps_thru-SM-Es) [m]'], ax=ax)
            df.plot(y=['cum snow mass change [m]', 'snow mass error [m]'], style = ['-.','-'], color = ['k', 'grey'], ax = ax)
        ax.set_ylabel('CumFlux [m]')
        ax.legend(loc='upper left', fontsize = 10, bbox_to_anchor = (1.0,1.1), frameon = False)
        max_error = max(abs(df['snow mass error [m]'].max()), abs(df['snow mass error [m]'].min()))
        ax.set_title(f'max error = {max_error*1000:.2f} mm')
        ax.set_xlim(df.index[0], df.index[-1])  
        ax.set_xlabel('')
    
    if UTC_time is not None:
        df.index = df.index.shift(UTC_time, freq = 'H')         
    
    if out_file is not None:
        df.to_csv(out_file)
    
    if resample_freq is not None:
        return df, df_rs
    else:
        return df

def load_output(model_dir, WB_filename, timestep = 'D', origin_date =
                '1980-01-01', **kwargs):
    """read ATS output files
    Parameters:
        model_dir: str
            ATS model directory
        WB_filename: str
            Waterbalance output filename, e.g. "water_balance.dat"
        timestep: str
            timestep string, 'D', 'H'
        origin_date: str,
            origin of model time
    Returns:
        datetime, data, and dataframe(datetime, data)
    
    """
    df = pd.read_csv(model_dir + WB_filename, comment='#')
    if timestep == 'D':
        # get datetime, convert seconds to days
        datetime_noLeap = rmLeapDays(df['time [d]'], freq='D', origin_date =
                                     origin_date, **kwargs)
    elif timestep == 'H':
        datetime_noLeap = rmLeapDays(df['time [h]'], freq='H', origin_date =
                                     origin_date, **kwargs)
    else:
        raise ValueError(f"Freq: {timestep} is not supported!")
    df['datetime'] = datetime_noLeap
    df.set_index('datetime', inplace=True, drop=True)
    
    return df



