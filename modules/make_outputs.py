# -*- coding: utf-8 -*-
"""
This script contains the functions used to make the excel file output for
the protein A bind and elute CADET model.
"""

import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def unpack_dicts(data, parent_key='', sep='_'):
    items = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(unpack_dicts(value, new_key, sep=sep))
        else:
            items[new_key] = [value]  # Wrap non-iterables in a list
    return items


def dict_to_df(data):
    unpacked_data = unpack_dicts(data)   
    # Create a DataFrame from the unpacked dictionary
    df = pd.DataFrame.from_dict(unpacked_data, orient='index').reset_index()
    df.columns = ['Key', 'Value']   
    return df


def convert_to_dataframe(item):
    if isinstance(item, pd.DataFrame):
        # Item is already a DataFrame, no conversion needed
        return item
    else:
        # Item is not a DataFrame, convert it
        try:
            df = pd.DataFrame(item)
            return df
        except:
            try:
               df = dict_to_df(item) 
               return df
            except Exception as e:
                print(f"Conversion to DataFrame failed: {e}")
                return None


def format_multicomponent(sim_solutions):    
    for sim_solution in sim_solutions:
        # unpack component concentration curves
        for i, c in enumerate(sim_solution['prot_c_mg']):
            sim_solution['component_' + str(i+1) + '_mg'] = c
        # delete combined concentration curve entry
        del sim_solution['prot_c_mM']
        del sim_solution['prot_c_mg']
    return sim_solutions
    

def make_results_folder(run_type, output_folder_path):
    path = os.path.join(output_folder_path, 'output')
    # Create the output folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    folder = str(run_type) + '_results_' + time.strftime("%Y%m%d%H%M%S")
    new_folder_path = os.path.join(path, folder)
    # Check if the "output" folder exists; if not, create it
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    os.mkdir(new_folder_path)                    
    return new_folder_path

 
def make_excel(all_results, folder_name, file_name, orientation='cols'):
    file_name = file_name + '.xlsx'
    file = os.path.join(folder_name, file_name)
    with pd.ExcelWriter(file) as writer:       
        for idx, df in enumerate(all_results):
            df = convert_to_dataframe(df)
            sheet_name = 'exp' + str(idx)
            if orientation == 'rows':
                df.T.to_excel(writer, sheet_name = sheet_name, header=False, index=True)
            else:
                df.to_excel(writer, sheet_name = sheet_name, header=True, index=False)
                
            
def make_excels(all_results, sim_solutions, folder):
    # Create excel with simulation results for each component in a sheet
    make_excel(all_results, folder, 'results', orientation='rows')
    # Create excel with simulation curves for each experiment in a sheet
    sim_solutions = format_multicomponent(sim_solutions)
    make_excel(sim_solutions, folder, 'curves', orientation='cols')
    
    
def convert_time_to_CV(time_data, ms):
    CV_data = np.zeros((len(time_data)))
    CV = ms['col_length']*ms['Ac_col']

    time_cumul = [0, ms['load_time'], ms['wash_time'], ms['elution_time'], time_data[-1]+1]
    flow_rate  = [ms['flow_rate'][0]] + ms['flow_rate'] + [ms['flow_rate'][-1]]

    for i in range(1, len(time_data)):
        for j in range(1, len(time_cumul)):
            if time_data[i] >= time_cumul[j-1] and time_data[i] < time_cumul[j]:            
                CV_data[i] = CV_data[i-1] + (time_data[i] - time_data[i-1])*(flow_rate[j])/CV 
    
    return CV_data


def make_plots(all_curves, ms_list, path):
    for i, ms in enumerate(ms_list):
        curves = all_curves[i]
        make_plot(curves, ms, path, i)
        

def make_plot(curves, ms, path, idx):
    path = os.path.join(path, f'exp_{idx}_column_sim_plot.png')    
    CVs = curves['CV']
    process_H = curves['H+']
    process_pH = curves['pH']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax2 = ax.twinx()
    
    if ms.get('outlet_H_vol_data') is not None:
        input_CV = ms['outlet_H_vol_data']/(ms['col_length']*ms['Ac_col'])
        input_pH = -np.log10(ms['H_data']*0.001)
        ax2.plot(input_CV, input_pH, linestyle='None', marker='o', markersize=3, color='b', label='input pH')
 
    elif ms.get('column_out') is not None:
        column_HUV_CV = ms['Ee']+(1-ms['Ee'])*ms['Ep']
        elu_tubing_HUV_CV = (ms['elu_tubing_length'] * np.pi*(ms['elu_tubing_id']*0.5)**2)/ms['col_vol']
        input_CV = ms['column_out']['CV'] + ms['load_CV'] + ms['wash_CV'] + column_HUV_CV + elu_tubing_HUV_CV
        input_pH = ms['column_out']['pH']
        ax2.plot(input_CV, input_pH, linestyle='None', marker='o', markersize=3, color='b', label='input pH')
   
    x_axis = CVs
    
    for idx, key in enumerate(curves.keys()):
        if key.startswith('component'):
            comp_mg = curves[key]
            name = f'protein {int(re.search(r'\d+', key).group())}'
            ax.plot(CVs, comp_mg, linewidth=2, color='k', label=name)
    
    # ax2.plot(x_axis, process_H, linewidth=2, color='r')
    ax2.plot(x_axis, process_pH, linewidth=2, color='r', label='sim pH')
       
    fig.legend(bbox_to_anchor=(1.35, 1))
    
    ax.set_xlim(ms['plot_start_CV'], ms['plot_end_CV'])
    
    ax.set_xlabel('Column volumes')
    ax.set_ylabel('Concentration [g/L]')   
    ax.tick_params(direction="in")    
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black') 

    ax2.set_ylabel('pH')        
    ax2.tick_params(direction="in")    
    ax2.tick_params(axis='y', colors='red')   
    ax2.yaxis.label.set_color('red')
    ax2.set_ylim(3, 9)
    
    fig.tight_layout()   
    plt.savefig(path, dpi=300, bbox_inches='tight')  
       
    