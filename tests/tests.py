#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests for proA model
"""
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
sys.path.append(os.path.join(os.getcwd(), "../modules"))
import create_model_structure, CADET_Process_configuration, evaluate_sim, make_outputs
import CSTR_v240927 as CSTR
import resin_titration_diffeq_v240927 as titration



def get_inputs(profile='step'):
    inputs = Dict()
    fit = Dict()
    pH_sim = Dict()

    inputs.run_type = 'forward'
    
    inputs.pool_start_conc = 5
    inputs.pool_end_conc   = 5
    inputs.plot_start_CV = 0
    inputs.plot_end_CV = 63
    inputs.load_CV = [38.24]
    inputs.wash_CV = [20]
    inputs.elution_CV = [5]
    inputs.load_RT = [2]
    inputs.wash_RT = [2]
    inputs.elution_RT = [2]
    inputs.load_pH = [7.4]
    inputs.wash_pH = [7.4]
    inputs.elution_pH = [3.5]
    inputs.molecular_weights = [[148,],]
    inputs.feed_conc = [[4.69,],]
    inputs.Ee = 0.4 #0.34
    inputs.Ep = 0.92
    inputs.particle_diameter = 54.1
    inputs.phase_ratio = 5.91e8
    inputs.col_id = 0.5
    inputs.col_length = 1.8
    inputs.load_tubing_id = 0.25
    inputs.load_tubing_length = 0
    inputs.elu_tubing_id = 0.25
    inputs.elu_tubing_length = 0
    inputs.params.pore_diff = [1.54]
    inputs.params.lnKe = [39.881]
    inputs.params.Bpp = [25.082]
    inputs.params.Ds = [0]
    inputs.params.lnKe0 = [39.91]
    inputs.params.lnKe1 = [-28.543]
    inputs.params.Bpp0 = [8.7894]
    inputs.params.Bpp1 = [-0.099]
    inputs.params.Ds0 = [0.00051]
    inputs.params.Ds1 = [93.031]
    
    pH_sim.mixer_volume = 0.09
    pH_sim.wash_buffer.buffer1_total = 50
    pH_sim.wash_buffer.buffer2_total = 0
    pH_sim.wash_buffer.NaCl = 150
    pH_sim.wash_buffer.Na = 150
    pH_sim.wash_buffer.Cl = 193
    pH_sim.elution_buffer.buffer1_total = 0
    pH_sim.elution_buffer.buffer2_total = 50
    pH_sim.elution_buffer.NaCl = 50
    pH_sim.elution_buffer.Na = 53.53
    pH_sim.elution_buffer.Cl = 50
    pH_sim.N = 60
    pH_sim.ligand_density = 20 
    pH_sim.ligand_pK = 4.0
    pH_sim.wash_buffer.pKa = 8.07
    pH_sim.wash_buffer.zA = 1
    pH_sim.elution_buffer.pKa = 4.76
    pH_sim.elution_buffer.zA = 0
    
    profile_path = r'example_inlet_profile.xlsx'
    
    fit.data_path = r'example_chromatogram.xlsx'
    fit.conc_cutoff = [1000]
    fit.bounds.pore_diff = ((1, 6),)
    fit.bounds.lnKe = ((25, 30),)
    fit.bounds.Bpp = ((5, 14),)
    fit.bounds.Ds = ((0.01, 2),)
    
    fit.bounds.lnKe0 = ((30, 50),)
    fit.bounds.lnKe1 = ((-30, -25),)
    fit.bounds.Bpp0 = ((5, 14),)
    fit.bounds.Bpp1 = ((-0.5, -0.001),)
    fit.bounds.Ds0 = ((1e-5, 2),)
    fit.bounds.Ds1 = ((10, 100),) 
    
    if inputs.run_type == 'fit':
        inputs.update(fit)
        
    if profile == 'simulated':
        inputs.update(pH_sim)
    
    elif profile == 'user_defined':
        inputs.profile_path = profile_path
        
    return inputs


def run_test(subprocess, profile, use_pH_dependence, folder):
    inputs = get_inputs(profile)
    ms_list = create_model_structure.create_model_structures(inputs, folder)
    
    if profile == 'simulated':
        for ms in ms_list:
            ms = CSTR.get_CSTR_profile(ms)
            CSTR.plot_CSTR(ms, folder)
            # CSTR.make_excel(ms, folder)
            
    if profile == 'simulated':
        for ms in ms_list:
            ms = titration.resin_titration(ms)
            titration.plot_titration(ms, folder)
            # titration.make_excel(ms, folder)
            
    for ms in ms_list:    
        sim_solution = CADET_Process_configuration.run(ms, subprocess=subprocess, profile=profile,
                                                       use_pH_dependence=use_pH_dependence,
                                                    )
        
        simulation_curves = evaluate_sim.process_simulation_data(sim_solution, ms)
        results = evaluate_sim.evaluate_simulation(simulation_curves, ms) 
        
        test_plot(simulation_curves, ms, subprocess, profile, use_pH_dependence, folder)
    

def test_plot(curves, ms, subprocess, profile, use_pH_dependence, folder):    
    path = os.path.join(folder, f'test_{subprocess}_{profile}_{use_pH_dependence}.png')    
    CVs = curves['CV']
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
    
    # unpack component concentration curves
    for i, c in enumerate(curves['prot_c_mg']):
        curves['component_' + str(i+1) + '_mg'] = c
    # delete combined concentration curve entry
    del curves['prot_c_mM']
    del curves['prot_c_mg']
    
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


###############################################################################
if __name__ == '__main__':     
    folder = make_outputs.make_results_folder(os.path.join('test'), os.getcwd())
    
    # run BTC tests
    run_test('BTC', 'step', False, folder)
    run_test('BTC', 'step', True, folder)
    
    # run LWE tests
    run_test('LWE', 'step', True, folder)
    run_test('LWE', 'linear', True, folder)
    run_test('LWE', 'user_defined', True, folder)
    run_test('LWE', 'simulated', True, folder)
    