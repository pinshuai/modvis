"""Functions used for generating ATS xml input files.

Authors: Ethan Coon, Pin Shuai
"""

import os,datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

import ats_input_spec
import ats_input_spec.public
import ats_input_spec.io

import amanzi_xml.utils.io as aio
import amanzi_xml.utils.search as asearch
import amanzi_xml.utils.errors as aerrors

def add_domains(main_list, mesh_filename, surface_region='surface', has_canopy=True):
    """add the subsurface and surface domains. Note this also adds 
    a 'computational domain' region to the region list, and a vis spec for domain."""
    # add subsurface domain
    ats_input_spec.public.add_domain(main_list, 
                                 domain_name='domain', 
                                 dimension=3, 
                                 mesh_type='read mesh file',
                                 mesh_args={'file':mesh_filename})
    
    # if surface_region:
    main_list['mesh']['domain']['build columns from set'] = surface_region    

    # Note this also adds a "surface domain" region to the region list and a vis spec for 
    # "surface"
    ats_input_spec.public.add_domain(main_list,
                            domain_name='surface',
                            dimension=2,
                            mesh_type='surface',
                            mesh_args={'surface sideset name':'surface'})
    # add snow domain
    ats_input_spec.public.add_domain(main_list,
                            domain_name='snow',
                            dimension=2,
                            mesh_type='aliased',
                            mesh_args={'target':'surface'})
    if has_canopy:
        # add canopy domain
        ats_input_spec.public.add_domain(main_list,
                            domain_name='canopy',
                            dimension=2,
                            mesh_type='aliased',
                            mesh_args={'target':'surface'})

def add_land_cover(main_list, nlcd_labels):
    """write a land-cover section for each NLCD type"""
    for nlcd_name in nlcd_labels:
        # this will load default values instead of pulling from the template
        ats_input_spec.public.set_land_cover_default_constants(main_list, nlcd_name)

    land_cover_list = main_list['state']['initial conditions']['land cover types']
    
    # update some defaults

    # note, these are from the CLM Technical Note v4.5
    #
    # Rooting depth curves from CLM TN 4.5 table 8.3
    #
    # Note, the mafic potential values are likely pretty bad for the types of van Genuchten 
    # curves we are using (ETC -- add paper citation about this topic).  Likely they need
    # to be modified.  Note that these values are in [mm] from CLM TN 4.5 table 8.1, so the 
    # factor of 10 converts to [Pa]
    #
    # Note, albedo of canopy taken from CLM TN 4.5 table 3.1
    def name_has_string(name, str_list):
        return any(s in name for s in str_list)
    
    for ilc in nlcd_labels:
        new_ilc = ilc.lower()
        if name_has_string(new_ilc, ["evergreen", "woody savannas"]):
            land_cover_list[ilc]['rooting profile alpha [-]'] = 7.0
            land_cover_list[ilc]['rooting profile beta [-]'] = 2.0
            land_cover_list[ilc]['rooting depth max [m]'] = 2.0
            land_cover_list[ilc]['mafic potential at fully closed stomata [Pa]'] = 2500785
            land_cover_list[ilc]['mafic potential at fully open stomata [Pa]'] = 647262  
        elif name_has_string(new_ilc, ["deciduous", "savannas", "mix"]):
            land_cover_list[ilc]['rooting profile alpha [-]'] = 6.0
            land_cover_list[ilc]['rooting profile beta [-]'] = 2.0
            land_cover_list[ilc]['rooting depth max [m]'] = 2.0
            land_cover_list[ilc]['mafic potential at fully closed stomata [Pa]'] = 2196768
            land_cover_list[ilc]['mafic potential at fully open stomata [Pa]'] = 343245
        elif name_has_string(new_ilc, ["shrub", "grassland"]):
            land_cover_list[ilc]['rooting profile alpha [-]'] = 7.0
            land_cover_list[ilc]['rooting profile beta [-]'] = 1.5
            land_cover_list[ilc]['rooting depth max [m]'] = 0.5
            land_cover_list[ilc]['mafic potential at fully closed stomata [Pa]'] = 4197396
            land_cover_list[ilc]['mafic potential at fully open stomata [Pa]'] = 813981
        else:
            logging.info(f"Default values are used for {ilc}!")

def soil_set_name(ats_id, subsurface_props):
    """add soil sets: note we need a way to name the set, so we use, e.g. SSURGO-MUKEY."""
    if ats_id == 999:
        return 'bedrock'
    source = subsurface_props.loc[ats_id]['source']
    native_id = subsurface_props.loc[ats_id]['native_index']
    if type(native_id) in [tuple,list]:
        native_id = native_id[0]
    elif type(native_id) is str:
        native_id = native_id.replace('(', '').replace(')', '').split(',')[0]
    else:
        raise("native_id is not a known type!")
    return f"{source}-{native_id}"

def get_main(config, subsurface_props, nlcd_labels, labeled_sets={}, side_sets={}, subcatchment_labels=None):
    """
    Get an ATS "main" input spec list -- note, this is a dummy and is not used to write any files yet

    """
    # get the main input structures including mesh, region, cycle driver, PKs, state, observations, and checkpoint
    main_list = ats_input_spec.public.get_main()
    
    # get PKs
    flow_pk = ats_input_spec.public.add_leaf_pk(main_list, 'flow', main_list['cycle driver']['PK tree'], 
                                            'richards-spec')

    # add the mesh and all domains
    # mesh_filename = os.path.join('..', config['mesh_filename'])
    mesh_filename = config['mesh_filename']
    add_domains(main_list, mesh_filename)
    
    # add labeled sets and sidesets
    try:
        for iname,ival in labeled_sets.items():
            ats_input_spec.public.add_region_labeled_set(main_list, iname, ival['setid'], mesh_filename, ival['entity'])
        for iname,ival in side_sets.items():
            ats_input_spec.public.add_region_labeled_set(main_list, iname, ival['setid'], mesh_filename, 'FACE')
    except:
        logging.info("no sidesets provided. adding surface and bottom only")
        for iname,id in zip(['surface','bottom'], [2,1]):
            ats_input_spec.public.add_region_labeled_set(main_list, iname, id, mesh_filename, 'FACE') 

    # add land cover
    add_land_cover(main_list, nlcd_labels)

    # add LAIs
    ats_input_spec.public.add_lai_evaluators(main_list, config['LAI_filename'], nlcd_labels)
    
    # add soil material ID regions, porosity, permeability, and WRMs
    for ats_id in subsurface_props.index:
        props = subsurface_props.loc[ats_id]
        set_name = soil_set_name(ats_id, subsurface_props)
        
        if props['van Genuchten n [-]'] < 1.5:
            smoothing_interval = 0.01
        else:
            smoothing_interval = 0.0
        
        ats_input_spec.public.add_soil_type(main_list, set_name, ats_id, mesh_filename,
                                            float(props['porosity [-]']),
                                            float(props['permeability [m^2]']), 
                                            1.e-9, # pore compressibility, maybe too large?
                                            float(props['van Genuchten alpha [Pa^-1]']),
                                            float(props['van Genuchten n [-]']),
                                            float(props['residual saturation [-]']),
                                            float(smoothing_interval))
        
    # add observations for each subcatchment for transient runs
    # this will add default observed variables instead of getting those from template
    
    obs = ats_input_spec.public.add_observations_water_balance(main_list, region="computational domain", 
                                                               surface_region= "surface domain")
    
    if subcatchment_labels is not None:
        for region in subcatchment_labels:
            obs = ats_input_spec.public.add_observations_water_balance(main_list, region, 
                                                                 outlet_region = region + ' outlet')
    return main_list

def populate_basic_properties(template_xml, main_xml, homogeneous_wrm=False, homogeneous_poro=False, homogeneous_perm=False):
    """This function updates an xml object with the above properties for mesh, regions, soil props, and lc props"""
    # find and replace the mesh list
    mesh_i = next(i for (i,el) in enumerate(template_xml) if el.get('name') == 'mesh')
    template_xml[mesh_i] = asearch.child_by_name(main_xml, 'mesh')

    # find and replace the regions list
    region_i = next(i for (i,el) in enumerate(template_xml) if el.get('name') == 'regions')
    template_xml[region_i] = asearch.child_by_name(main_xml, 'regions')
    
    # find and replace land cover
    consts_list = asearch.find_path(template_xml, ['state', 'initial conditions'])
    try:
        lc_i = next(i for (i,el) in enumerate(consts_list) if el.get('name') == 'land cover types')
    except StopIteration:
        pass
    else:
        consts_list[lc_i] = asearch.find_path(main_xml, ['state', 'initial conditions', 'land cover types'])
        
    # find and replace the WRMs list -- note here we only replace the inner "WRM parameters" because the
    # demo has this in the PK, not in the evaluators list
    if not homogeneous_wrm:
        wrm_list = asearch.find_path(template_xml, ['PKs', 'water retention evaluator'])
        wrm_i = next(i for (i,el) in enumerate(wrm_list) if el.get('name') == 'WRM parameters')
        wrm_list[wrm_i] = asearch.find_path(main_xml, ['PKs','water retention evaluator','WRM parameters'])

    fe_list = asearch.find_path(template_xml, ['state', 'evaluators'])

    # update LAIs in the template
    # consts_list = asearch.find_path(template_xml, ['state', 'initial conditions'])
    try:
        lc_i = next(i for (i,el) in enumerate(fe_list) if el.get('name') == 'canopy-leaf_area_index')
    except StopIteration:
        pass
    else:    
        fe_list[lc_i] = asearch.find_path(main_xml, ['state', 'evaluators', 'canopy-leaf_area_index'])    
    
    # find and replace porosity, permeability
    if not homogeneous_poro:
        poro_i = next(i for (i,el) in enumerate(fe_list) if el.get('name') == 'base_porosity')
        fe_list[poro_i] = asearch.find_path(main_xml, ['state', 'evaluators', 'base_porosity'])

    if not homogeneous_perm:
        perm_i = next(i for (i,el) in enumerate(fe_list) if el.get('name') == 'permeability')
        fe_list[perm_i] = asearch.find_path(main_xml, ['state', 'evaluators', 'permeability'])

def create_unique_name(name, homogeneous_wrm=False, homogeneous_poro=False, homogeneous_perm=False):
    suffix = '_h'
    if homogeneous_perm:
        suffix += 'K'
    if homogeneous_poro:
        suffix += 'p'
    if homogeneous_wrm:
        suffix += 'w'
    if suffix == '_h':
        suffix = ''
    return name + suffix
        
def write_spinup_steadystate(config, main_xml, mean_precip=1e-8, **kwargs):
    """ Write the spinup steadystate xml file.
    
    Parameters:
        config: dict,
            Model configuration dictionary.
        main_xml: xml object,
            The main xml object.
        mean_precip: float,
            Mean annual precipitation. Default is 1e-8 m/s
    """
    # name = create_unique_name(name, **kwargs)
    template_file = config['spinup_steadystate_template']
    filename = config[f'spinup_steadystate_xml']
    logging.info(f'Writing spinup steadystate: {filename}')
    
    # load the template file
    template_xml = aio.fromFile(template_file)
    
    # populate basic properties for mesh, regions, and soil properties
    populate_basic_properties(template_xml, main_xml, **kwargs)

    # set the mean avg source as 60% of mean precip
    precip_el = asearch.find_path(template_xml, ['state', 'evaluators', 'surface-precipitation', 
                                        'function-constant', 'value'])
    precip_el.setValue(mean_precip * .6)
   
    # write to disk
    aio.toFile(template_xml, config[f'spinup_steadystate_xml'])

    # make a run directory
    try:
        os.mkdir(os.path.join('..', '..', 'model', config[f'spinup_steadystate_rundir']))
    except FileExistsError:
        pass

def write_transient(config, main_xml, start_date, end_date, cyclic_steadystate=False,
                    time0 = "1980-1-1", **kwargs):
    """Write transient xml file using template. 
    
    Parameters:
        config: dict,
            Model configuration dictionary.
        start_date: str,
            Start date of the transient run. Note it defaults to '1980-10-1' for cyclic runs.
        end_date: str,
            End date of the transient run. Note it defaults to '1990-10-1' for cyclic runs.  
        cyclic_steadystate: bool
            Generate input xml for cyclic runs if True. Default is False.
        time0: str,
            Default origin time in the model. This should be consistent with all other input files 
            including forcing and LAI.
            
    Returns:
        None
    """ 

    if cyclic_steadystate:
        prefix = 'spinup_cyclic'
        # start_year = 1980
        # end_year = 1990
        start_datetime = datetime.datetime.strptime("1980-10-1", '%Y-%m-%d').date()
        end_datetime = datetime.datetime.strptime("1990-10-1", '%Y-%m-%d').date()        
        previous = 'spinup_steadystate'
        template_filename = config['spinup_cyclic_template']

    else:
        prefix = 'transient'
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        previous = 'spinup_cyclic'
        template_filename = config['transient_template']
        
    filename = config[f'{prefix}_xml']
    logging.info(f'Writing {prefix} xml: {filename}')
    # template_filename = template_dir + f'{prefix}-template.xml'
    
    # load the template file
    template_xml = aio.fromFile(template_filename)

    # populate basic properties for mesh, regions, and soil properties
    populate_basic_properties(template_xml, main_xml, **kwargs)

    # update the DayMet filenames
    # wind speed uses default?
    if cyclic_steadystate:
        daymet_filename = config['daymet_typical_filename']
        LAI_filename = config['LAI_typical_filename']
    else:
        daymet_filename = config['daymet_filename']
        LAI_filename = config['LAI_filename']
        
    for var in ['surface-incoming_shortwave_radiation',
                'surface-precipitation_rain',
                'snow-precipitation',
                'surface-air_temperature',
                'surface-vapor_pressure_air',
                'surface-temperature',
                ]:
        try:
            par = asearch.find_path(template_xml, ['state', 'evaluators', var, 'file'])
        except aerrors.MissingXMLError:
            pass
        else:
            par.setValue(daymet_filename)
    
    # update the LAI filenames
    for par in asearch.findall_path(template_xml, ['canopy-leaf_area_index', 'file']):
        par.setValue(os.path.join(LAI_filename))
    
    # update the start and end time -- start at Oct 1 of year 0, end 10 years later

    origin_datetime = datetime.datetime.strptime(time0, '%Y-%m-%d').date()
    start_days = (start_datetime - origin_datetime).total_seconds() // 86400
    end_days = (end_datetime - origin_datetime).total_seconds() // 86400
    
    # if start_day is None:
    #     start_day = 274 + 365*(start_year - 1980)
    par = asearch.find_path(template_xml, ['cycle driver', 'start time'])
    par.setValue(start_days)

    # if end_day is None:
    #     end_day = 274 + 365*(end_year - 1980)
    par = asearch.find_path(template_xml, ['cycle driver', 'end time'])
    par.setValue(end_days)
    
    # update the restart filenames
    for var in asearch.findall_path(template_xml, ['initial condition', 'restart file']):
        var.setValue(os.path.join('..', config[f'{previous}_rundir'], 'checkpoint_final.h5'))

    # update the observations list
    obs = next(i for (i,el) in enumerate(template_xml) if el.get('name') == 'observations')
    template_xml[obs] = asearch.child_by_name(main_xml, 'observations')
   
    # update surface-incident-shortwave-radiation
    par = asearch.find_path(template_xml, ['state', 'evaluators', 'surface-incident_shortwave_radiation', 'latitude [degrees]'])
    par.setValue(config['latitude [deg]'])   
    
    # write to disk and make a directory for running the run
    filename = config[f'{prefix}_xml']
    aio.toFile(template_xml, filename)
    # rundir = config[f'{prefix}_{name}_rundir']

    
    try:
        os.mkdir(os.path.join('..', '..', 'model', config[f'{prefix}_rundir']))
    except FileExistsError:
        pass


