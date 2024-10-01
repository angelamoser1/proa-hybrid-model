# -*- coding: utf-8 -*-
"""
This script contains the functions used to process the simulation results for
the protein A bind and elute CADET model.

Note, the inlet profile is pulled from the front of the column, so we only need
to account for column hold up volume, not tubing etc., between the inlet and
outlet profiles.
"""
import numpy as np
import math
import chrom_eval_tools


def process_simulation_data(simulation_results, ms):    
    # time = simulation_results.solution.column.outlet.time
    # solution_array = simulation_results.solution.column.outlet.solution
    # inlet_array = simulation_results.solution.column.inlet.solution
    
    time = simulation_results.solution.outlet.outlet.time
    solution_array = simulation_results.solution.outlet.outlet.solution
    inlet_array = simulation_results.solution.outlet.inlet.solution

    ms['modifier'] = 'pH'
    
    inlet = {}
    s = {}
    s['time_sec'] = time # [s]
        
    if ms.get('modifier') == None:
        inlet_component_c = inlet_array.T
        component_c = solution_array.T
        inlet_component_c_mg = inlet_component_c * np.array(ms['MW']).reshape(len(ms['MW']), 1)
        component_c_mg = component_c * np.array(ms['MW']).reshape(len(ms['MW']), 1)
    else:
        if ms.get('modifier') == 'pH':     
            s['H+'] = solution_array[:,0] # [mM]
            s['pH'] = np.array([-math.log10(.001*c) for c in s['H+']])
        elif ms.get('modifier') == 'salt':
            s['salt'] = solution_array[:,0] # [mM]
        inlet_component_c = inlet_array[:,1:].T
        component_c = solution_array[:,1:].T
        inlet_component_c_mg = inlet_component_c * np.array(ms['MW'][1:]).reshape(len(ms['MW'])-1, 1)
        component_c_mg = component_c * np.array(ms['MW'][1:]).reshape(len(ms['MW'])-1, 1)
             
    inlet['prot_c_mM'] = inlet_component_c
    s['prot_c_mM'] = component_c
    inlet['prot_c_mg'] = inlet_component_c_mg
    s['prot_c_mg'] = component_c_mg

    # Collect time section parameters & flow rate
    time_cumul = [0, ms['load_time'], ms['wash_time'], ms['elution_time'], s['time_sec'][-1]+1]
    flow_rate  = [ms['flow_rate'][0]] + ms['flow_rate'] + [ms['flow_rate'][-1]]
            
    # Create a volume column [mL] from time column [s] using flow rate
    s['volume'] = np.zeros((len(s['time_sec'])))
    for i in range(1, len(s['time_sec'])):
        for j in range(1, len(time_cumul)):
            if s['time_sec'][i] >= time_cumul[j-1] and s['time_sec'][i] < time_cumul[j]:            
                s['volume'][i] = s['volume'][i-1] + (s['time_sec'][i] - s['time_sec'][i-1])*flow_rate[j]*1e6                  

    # Convert from time (s) to time (min)
    s['time_min'] = s['time_sec']/60
    
    # Convert from volume [mL] to CV
    s['CV'] = s['volume']/(ms['col_vol']*1e6)
    
    # calculate cumulative purity and yield for the simulation         
    s['purity'], s['yield'] = chrom_eval_tools.purity_and_yield(inlet['prot_c_mg'], s['prot_c_mg'])

    return s
 

def same_dimensions(x_data, y_data):
    x_dim = np.ndim(x_data)
    y_dim = np.ndim(y_data)
    
    if x_dim == y_dim:
        return x_data, y_data

    if x_dim == 1 and y_dim == 2:
        # Handle case where x_data is 1D and y_data is 2D
        new_x_data = np.array([x_data] * len(y_data))
        new_y_data = np.array(y_data)
        return new_x_data, new_y_data

    # If dimensions are not the same and not the case handled above
    raise ValueError("Incompatible dimensions: x_data and y_data cannot be broadcasted together.")
    

def select_section(start_cutoff, end_cutoff, x_data, y_data):
    def get_section_indices(start_cutoff, end_cutoff, x_data):
        # Find the indices corresponding to the cutoffs
        start_idx = np.searchsorted(x_data, start_cutoff, side='left')
        end_idx = np.searchsorted(x_data, end_cutoff, side='right')
        return start_idx, end_idx

    if x_data.ndim == 1:
        # Handle 1D case
        start_idx, end_idx = get_section_indices(start_cutoff, end_cutoff, x_data)
        x_section = x_data[start_idx:end_idx]
        y_section = y_data[start_idx:end_idx]
    else:
        # Handle multi-dimensional case
        x_section = []
        y_section = []
        for i in range(len(x_data)):
            start_idx, end_idx = get_section_indices(start_cutoff, end_cutoff, x_data[i])
            x_section.append(x_data[i][start_idx:end_idx])
            y_section.append(y_data[i][start_idx:end_idx])

    return np.array(x_section), np.array(y_section)
        
    
# s is the processed simulation solution as a dictionary
def evaluate_simulation(s, ms):
    # select data from the BTC section
    vol_sim, conc_sim = same_dimensions(s['volume'], s['prot_c_mg'])
    load_vol_sim, load_conc_sim = select_section(0, ms['load_vol']*1e6, vol_sim, conc_sim)
    DBC10 = []
    EBC = []
    for idx, comp_feed_c in enumerate(ms['feed_conc_mg']):
        load_v = load_vol_sim[idx] - ms['HUV']
        load_c = load_conc_sim[idx]
        # calculate DBC10
        _ , DBC_comp = chrom_eval_tools.calculate_DBC(load_v, load_c,
                                                  comp_feed_c,
                                                  CV=ms['col_vol']*1e6,
                                                  end_breakthrough=load_v[-1])
        DBC10.append(DBC_comp)         
    
        # calculate EBC
        _ , EBC_comp = chrom_eval_tools.calculate_EBC(load_v, load_c,
                                                  comp_feed_c,
                                                  CV=ms['col_vol']*1e6,
                                                  end_breakthrough=load_v[-1])
        EBC.append(EBC_comp)

    # Select data from the elution section
    elution_vol, elution_conc = select_section(ms['wash_vol']*1e6,
                                                     ms['elution_vol']*1e6,
                                                     vol_sim,
                                                     conc_sim)
     
    # evaluate the elution peak
    pool_volume, pool_conc = chrom_eval_tools.evaluate_elution(elution_vol[0],
                                                               elution_conc,
                                                               ms['pool_start_conc'],
                                                               ms['pool_end_conc'])
              
    # Convert pool volume to CV
    pool_volume = [pool_volume/(ms['col_vol']*1e6)] * len(ms['feed_conc_mg'])
        
    # Calculate recovery
    mass_bal = []
    for idx, comp_feed_c in enumerate(ms['feed_conc_mg']):
        load_mass = comp_feed_c*ms['load_vol']*1e6
        total_mass = np.trapz(s['prot_c_mg'][idx], s['volume'])    
        mass_bal_comp = 100*total_mass/load_mass
        mass_bal.append(mass_bal_comp)
        
    # save results in dictionary   
    results = {'DBC10':DBC10, 'EBC':EBC, 'pool vol':pool_volume,
               'pool C':pool_conc, 'mass_bal':mass_bal}
    
    results.update(ms['params'])
    
    return results