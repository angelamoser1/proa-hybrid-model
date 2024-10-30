# -*- coding: utf-8 -*-
"""
This script contains the functions used to create the dictionary (ms) used to 
pass information within the protein A bind and elute CADET model.

Based on user input into the data_entry script, a list of model structures
(dictionaries) is created, with one dictionary corresponding to each experiment.
"""
import numpy as np
import pandas as pd
import transport
import make_outputs


def create_model_structures(inputs, folder):
    # initialize list for all model structures            
    model_structures = []
        
    # loop over experiments and create one ms for each
    for experiment, concs in enumerate(inputs['feed_conc']):
        # create ms
        ms = create_ms(inputs, experiment)
        # add ms for this experiment to the list of ms for all experiments        
        model_structures.append(ms)
        
    # save model structure inputs in an excel file
    make_outputs.make_excel([inputs], folder, 'model_inputs', orientation='cols')      
    
    # read in chromatogram data if provided
    if inputs.get('data_path') is not None:
        # pull in experimental curve data from excel file
        file = inputs['data_path']   
        # Read all sheets into a dictionary of DataFrames
        chromatogram_data = pd.read_excel(file, sheet_name=None)
                 
        # loop over experiments (excel sheets)
        for experiment, sheet_data in enumerate(chromatogram_data.items()):
            ms = model_structures[experiment]
            # Unpack the tuple
            _, sheet_df = sheet_data

            # separate concentration and volume data for chromatograms
            conc_vol_data = []
            conc_time_data = []
            conc_data = []
            for idx, column in enumerate(sheet_df.columns):
                column = sheet_df[column].astype(float).to_numpy()
                column = column[~np.isnan(column)]
                if idx % 2 == 0:
                    conc_vol_data.append(column)
                else:
                    conc_data.append(column)
            
            conc_data_mM = [[0 for _ in row] for row in conc_data]
            # column by column transformations
            for i in range(len(conc_vol_data)):
                v_data = conc_vol_data[i] * 1e-6  # convert from mL to m^3
                # trim off data before load starts and after elution ends
                mask = (v_data >= 0) & (v_data <= ms['elution_vol'])
                
                # Use the mask to filter both v_data and conc_data in place
                conc_vol_data[i] = v_data[mask]
                conc_data[i] = conc_data[i][mask]
                
                # convert to mM
                conc_data_mM[i] = conc_data[i]/ms['molecular_weights'][i]

                # convert volume data to time (s)
                time_col = convert_vol_to_time(conc_vol_data[i], ms)
                conc_time_data.append(time_col)

            # add concentration data to ms        
            ms['conc_vol_data'] = np.array(conc_vol_data)
            ms['conc_time_data'] = np.array(conc_time_data)
            ms['conc_data'] = np.array(conc_data)
            ms['conc_data_mM'] = np.array(conc_data_mM)

    # read in [H+] profile data if provided        
    if inputs.get('profile_path') is not None:
        # pull in experimental curve data from excel file
        file = inputs['profile_path']   
        # Read all sheets into a dictionary of DataFrames
        H_data = pd.read_excel(file, sheet_name=None, usecols=[0, 1])
        
        # loop over experiments (excel sheets)
        for experiment, sheet_data in enumerate(H_data.items()): 
            ms = model_structures[experiment]
            # Unpack the tuple
            _, sheet_df = sheet_data

            # pull in volume and [H+] data (convert mL to m^3)
            outlet_H_vol_data = np.array(sheet_df.iloc[:, 0])*1e-6
            
            # shift back trace by column HUV                      
            inlet_H_vol_data = adjust_for_HUV(outlet_H_vol_data, ms)
            
            # trim off data before load starts and after elution ends
            mask = (inlet_H_vol_data >= 0) & (inlet_H_vol_data <= ms['elution_vol'])
            inlet_H_vol_data = inlet_H_vol_data[mask]

            H_time_data = convert_vol_to_time(inlet_H_vol_data, ms)

            H_data = np.array(sheet_df.iloc[:, 1])   
            H_data = H_data[mask]

            # add concentration data to ms        
            ms['outlet_H_vol_data'] = np.array(outlet_H_vol_data)
            ms['inlet_H_vol_data'] = np.array(inlet_H_vol_data)
            ms['H_time_data'] = np.array(H_time_data)
            ms['H_data'] = np.array(H_data)
                     
    return model_structures


def adjust_for_HUV(vol_data, ms): 
    column_HUV = (ms['Ee']+(1-ms['Ee'])*ms['Ep'])*ms['col_length']*ms['Ac_col']
    load_HUV = column_HUV + ms['load_tubing_length'] * np.pi*(ms['load_tubing_id']*0.5)**2
    elu_HUV = column_HUV + ms['elu_tubing_length'] * np.pi*(ms['elu_tubing_id']*0.5)**2

    adj_vol_data = np.zeros((len(vol_data)))
    
    for i in range(0, len(vol_data)):
        if (vol_data[i] - load_HUV) < ms['wash_vol']:        
            adj_vol_data[i] = vol_data[i] - load_HUV
        else:
            adj_vol_data[i] = vol_data[i] - elu_HUV
          
    return adj_vol_data


def convert_vol_to_time(vol_data, ms): 
    '''convert full chromatogram section by section'''
    def convert_section(vol_data, flow_rate, t_start=0): 
        '''for a constant flow rate section'''
        # first subtract v_start from volume data so it starts at 0
        vol_data = vol_data - vol_data[0]
        # divide volume data by flow rate to get time 
        time_data = vol_data / flow_rate
        # add the start time back in
        time_data = time_data + t_start
        return time_data
    
    # Create empty array for time
    time_data = []

    section_vols = [0, ms['load_vol'], ms['wash_vol'], ms['elution_vol']] # m^3
    section_times = [0, ms['load_time'], ms['wash_time'], ms['elution_time']] # m^3
    flow_rate  = ms['flow_rate'] # m^3/s
    
    # loop over contant flow rate sections    
    for idx, f in enumerate(flow_rate):
        v_start = section_vols[idx]
        v_end = section_vols[idx + 1]
        t_start = section_times[idx]
        mask = (vol_data >= v_start) & (vol_data <= v_end)
        v_section = vol_data[mask]
        if len(v_section) > 0:
            t_section = convert_section(v_section, f, t_start=t_start)
            time_data.extend(t_section)
        else:
            continue
        
    time_data = np.array(time_data)
    
    return time_data


def select_experiment(inputs, experiment):
    exp = {}
    
    # copy all key/value pairs from inputs dict into exp
    exp.update(inputs)
    
    # UNIT CONVERSIONS (all to SI - m, mol, s)
    exp['col_id'] = exp['col_id']/100 # [m]
    exp['col_length'] = exp['col_length']/100 # [m]
    exp['Ac_col'] = 0.25*np.pi*exp['col_id']**2   
    exp['col_vol'] = exp['Ac_col']*exp['col_length'] # [m^3]
    
    # for inputs that can be different for each experiment
    # select only the values for the given experiment
       
    exp['feed_conc_mg'] = inputs['feed_conc'][experiment]
    exp['molecular_weights'] = inputs['molecular_weights'][experiment]     
    exp['feed_conc_mM'] = [c_mg / MW_comp for c_mg, MW_comp in zip(exp['feed_conc_mg'], exp['molecular_weights'])]
     
    exp['load_pH'] = inputs['load_pH'][experiment]
    exp['wash_pH'] = inputs['wash_pH'][experiment]
    exp['elution_pH'] = inputs['elution_pH'][experiment]
    
    exp['load_CV'] = inputs['load_CV'][experiment]
    exp['wash_CV'] = inputs['wash_CV'][experiment]
    exp['elution_CV'] = inputs['elution_CV'][experiment]
    
    exp['load_RT'] = inputs['load_RT'][experiment]
    exp['wash_RT'] = inputs['wash_RT'][experiment]
    exp['elution_RT'] = inputs['elution_RT'][experiment]
    
    # Calculate the end time for each section in the simulation [s]   
    exp['load_time'] = exp['load_RT']*60*exp['load_CV']
    exp['wash_time'] = exp['wash_RT']*60*exp['wash_CV'] + exp['load_time']
    exp['elution_time'] = exp['elution_RT']*60*exp['elution_CV'] + exp['wash_time']
    
    # Calculate the end volume for each section in the simulation [m^3]
    exp['load_vol'] = exp['col_vol']*exp['load_CV']
    exp['wash_vol'] = exp['col_vol']*exp['wash_CV'] + exp['load_vol']
    exp['elution_vol'] = exp['col_vol']*exp['elution_CV'] + exp['wash_vol']
    
    return exp


def create_ms(inputs, experiment):  
    # pull in experiment specific information
    exp = select_experiment(inputs, experiment)
    
    # create the model structure (dictionary) to contain data for passing into functions
    ms = {}
    
    # copy all key/value pairs from exp dict into ms
    ms.update(exp)
         
    # pull in values needed for further calculations and unit conversions    
    load_tubing_id = ms['load_tubing_id']                # [mm], tubing diameter
    load_tubing_length = ms['load_tubing_length']        # [cm], tubing length (injection valve to column) 
    elu_tubing_id = ms['elu_tubing_id']                  # [mm], tubing diameter
    elu_tubing_length = ms['elu_tubing_length']          # [cm], tubing length (injection valve to column)     
    particle_diameter = ms['particle_diameter']          # [um]
    
    Ee = ms['Ee']
    Ep = ms['Ep']                                  
    
#### Unit conversions and preliminary calcs ###################################  
    
    # UNIT CONVERSIONS (all to SI - m, mol, s)            
    load_tubing_length = load_tubing_length/100      # [m]
    load_tubing_id = load_tubing_id/1e3              # [m] 
    elu_tubing_length = elu_tubing_length/100        # [m]
    elu_tubing_id = elu_tubing_id/1e3                # [m] 
    particle_diameter = particle_diameter*1e-6       # [m]
    
    # update ms after unit conversions
    ms['load_tubing_id'] = load_tubing_id
    ms['load_tubing_length'] = load_tubing_length 
    ms['elu_tubing_id'] = elu_tubing_id
    ms['elu_tubing_length'] = elu_tubing_length 
    ms['particle_diameter'] = particle_diameter 

    # More calculations   
    load_tubing_vol = load_tubing_length*(0.25*np.pi*load_tubing_id**2)       # [m^3]
    residence_time =  [ms['load_RT'], ms['wash_RT'], ms['elution_RT']]
    flow_rate = [ms['col_vol'] / (RT*60) for RT in residence_time] # [m^3/s]
    col_lin_vel = [FR / ms['Ac_col'] for FR in flow_rate] # [m/s]
    col_int_vel = [i/Ee for i in col_lin_vel] # [m/s]                                                          
    Et = Ee + Ep*(1-Ee) # [-], Total porosity
    HUV = load_tubing_vol + ms['col_vol']*Et  # [m^3]
       
    # update ms with calculated values
    ms['Et'] = Et
    ms['flow_rate'] = flow_rate
    ms['col_lin_vel'] = col_lin_vel
    ms['col_int_vel'] = col_int_vel
    ms['HUV'] = HUV
    
    # Conversions for buffering sims
    if ms.get('ligand_density') is not None:
        ms['ligand_density'] = ms['ligand_density'] / (1 - ms['Et'])  # Convert from mM col to mM solid phase   
    if ms.get('ligand_pK') is not None:    
        ms['ligand_K'] = 10 ** -ms['ligand_pK']  # Ka of ligand
    if ms.get('mixer_volume') is not None:
        ms['mixer_volume'] = ms['mixer_volume']*1e-6 # [m^3]
    
    # Combine all molecular weights in list for further calculations     
    MW_H = 0.019 # [kDa], molecular weight of H3O+
    ms['MW'] = [MW_H] + ms['molecular_weights']

    ms = transport.calculate_transport(ms)
    
    # set defaults. These can be overridden by user input.
        
    if ms.get('coord_num') is None:
        ms['coord_num'] = 6 # [-], Coordination number (hexagonal lattice)
    
    if ms.get('kappa') is None:
        ms['kappa'] = 4e-9 # [m], Debye parameter
        
    if ms.get('k_kin_colloidal') is None:
        ms['k_kin_colloidal'] = 1e8 # [1/s] Kinetic rate constant for colloidal isotherm    
    
    if ms.get('zI') is None:
        ms['zI'] = 1 # Net charge of H+
    
    if ms.get('k_kin_titration') is None:
        ms['k_kin_titration'] = 1e13 # [1/s] Kinetic rate constant for LDF in resin titration
       
    if ms.get('column_axial_nodes') is None:
        ms['column_axial_nodes'] = 35
        
    if ms.get('particle_nodes') is None:
        ms['particle_nodes'] = 35
        
    if ms.get('tubing_axial_nodes') is None:
        ms['tubing_axial_nodes'] = 40
        
    if ms.get('n_threads') is None:
        ms['n_threads'] = 4
        
    if ms.get('abstol') is None:
        ms['abstol'] = 1e-10
    
    if ms.get('reltol') is None:
        ms['reltol'] = 1e-10
 
    return ms
