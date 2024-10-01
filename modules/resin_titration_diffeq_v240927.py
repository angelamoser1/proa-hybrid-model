#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to predict pH transients
Equations adapted from Pabst and Carta 2006 using Eqs 34-36.
This is the python version of the script created by Soumitra Bhoyar to model 
in-column pH gradients in protein A chromatography. 
which was modified from Scott Altern's original Matlab script.
Between loading and elution, we change buffer species (Tris, acetate), pH 
and ionic strengths.
Only Na+ ions 'adsorb' on the resin

This version uses the analytical solution to the quadratic electroneutrality.

Experimental data for overlay should have the start of the elution set as 0
All units should be mol/L, cm, L, min
"""
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def unpack_buffer(buffer_dict):
    pKa = buffer_dict['pKa']
    z = buffer_dict['zA']
    A = 0.5114
    b = 0.16
    return pKa, z, A, b


def initialize_arrays(N, cT0, cA0, cCl0, cNa0, qNa0):
    # initialize 2D numpy array with y vector as first column and dydt as second column
    # keeping them together should be faster for access, hopefully.
    big_array = np.zeros((5*N+5, 2))    
    # set y0 as the first column
    y0 = big_array[:,0]
    # set dydt as the second column
    dydt = big_array[:,1]    
    # Apply initial conditions to the axial cells; concentrations at t=0.
    y0[0*N:1*N+1] = cT0
    y0[1*N+1:2*N+2] = cA0
    y0[2*N+2:3*N+3] = cCl0
    y0[3*N+3:4*N+4] = cNa0
    y0[4*N+4:5*N+5] = qNa0
    
    # create another 2D array for iteratively calculated variables
    v0 = np.zeros((N+1, 4))
    
    return(y0, dydt, v0)
    

def resin_titration(ms): 
    # unpack
    pKa1, zA1, A1, b1 = unpack_buffer(ms['wash_buffer'])
    pKa2, zA2, A2, b2 = unpack_buffer(ms['elution_buffer'])
    zI = ms['zI']   
    K = ms['ligand_K']
    N = ms['N']

    # "Ligand" density - convert from mM SP to M solid phase
    qR = ms['ligand_density'] * 1e-3
    
    # get time points for inlet profile from CSTR output
    # convert from seconds to minutes
    t_eval = [t / 60 for t in ms['inlet_time']]

    # pull initial concentrations of each component from the output of the CSTR model
    pH = np.array(ms['cstr_out']['pH'])
    IS = np.array(ms['cstr_out']['ionic_strength'])
    cA_ion = np.array(ms['cstr_out']['Acetate_ion'])
    cAH = np.array(ms['cstr_out']['Acetic_acid'])
    cTris = np.array(ms['cstr_out']['Tris'])
    cTH = np.array(ms['cstr_out']['TrisH_ion'])  
    cCl = np.array(ms['component_array'][3])
    cNa = np.array(ms['component_array'][2])
    
    cT = cTris + cTH
    cA = cA_ion + cAH

    ms['cNa'] = cNa
    ms['cCl'] = cCl
    ms['cT_total'] = cT
    ms['cA_total'] = cA
    
    # Ionic strength dependent dissociation constant calculations
    Ka1I = equilib_constant(IS, A1, b1, zA1, pKa1)
    Ka2I = equilib_constant(IS, A2, b2, zA2, pKa2)

    # Get H+ conc using the quadratic electroneutrality equation
    cHI = get_cH_quadratic_neutrality(cNa, cCl, Ka1I, Ka2I, cA, cT) 
    
    # calculate activity coefficient and pH for plotting (troubleshooting)
    # gI = activity_coeff(IS, A1, b1, zI)
    # pHi = -np.log10(gI * cHI)
    # plt.plot(t_eval, pHi, label='pH after IS')

    # Solid phase (Adsorbed) Na+ conc. at region I (mol/L SP)
    qNa = 0.5 * (-K * cNa / cHI + np.sqrt((K * cNa / cHI)**2 + 4 * qR * K * cNa / cHI))   
    
    # Apply initial conditions to the axial cells; concentrations at t=0.
    cT0 = np.ones(N + 1) * cT[0]
    cA0 = np.ones(N + 1) * cA[0]
    cCl0 = np.ones(N + 1) * cCl[0]
    cNa0 = np.ones(N + 1) * cNa[0]
    qNa0 = np.ones(N + 1) * qNa[0]
    y0 = np.concatenate([cT0, cA0, cCl0, cNa0, qNa0])
    
    # -------------------------------------------------------------------------
    print('Resin titration simulation running...')
    tic = time.time()
    
    # Solving the model
    sol = solve_ivp(dsys, t_span=(0, t_eval[-1]), t_eval=t_eval, y0=y0, 
                    args=(ms,), rtol=2.3e-14, atol=1e-18, method='Radau') # atol 1e-17 gives smooth pH
    
    toc = time.time()
    elapsed_time = toc - tic
    print(f'elapsed time: {elapsed_time:.2f} seconds')
    # -------------------------------------------------------------------------
    # get time and solution vectors from solver
    t = sol.t
    y = sol.y.T

    # Parse column profiles
    cT = y[:, 0*N:1*N+1]  # Tris
    cA = y[:, 1*N+1:2*N+2]  # Acetate
    cCl = y[:, 2*N+2:3*N+3]  # Chloride
    cNa = y[:, 3*N+3:4*N+4]  # Liquid phase Na
    qNa = y[:, 4*N+4:5*N+5]  # Solid phase Na

    # Calculate Ka and pH (for each axial section)
    IS_profile = np.maximum(cCl, cNa) 
    Ka1 = equilib_constant(IS_profile, A1, b1, zA1, pKa1)
    Ka2 = equilib_constant(IS_profile, A2, b2, zA2, pKa2)
    g = activity_coeff(IS_profile, A1, b1, zI)

    # Quadratic electroneutrality equation for each axial section gives H+
    # concentration as an array of roots
    # each input is a 2D numpy array
    cH = get_cH_quadratic_neutrality(cNa, cCl, Ka1, Ka2, cA, cT)
    pH = np.real(-np.log10(g*cH))

    ms['pH_array'] = pH    
    ms['pH_in'] = pH[:, 0]
    
    # Assign outlet profiles
    ms['column_out'] = {}
    
    ms['column_out']['cT'] = cT[:, -1]
    ms['column_out']['cA'] = cA[:, -1]
    ms['column_out']['cCl'] = cCl[:, -1]
    ms['column_out']['cNa'] = cNa[:, -1]
    ms['column_out']['qNa'] = qNa[:, -1]
    ms['column_out']['cH'] = cH[:, -1]
    ms['column_out']['pH'] = pH[:, -1]
    ms['column_out']['cTH'] = cT[:, -1] * cH[:, -1] / (cH[:, -1] + Ka2[:, -1])
    ms['column_out']['cA_ion'] = Ka1[:, -1] * cA[:, -1] / (cH[:, -1] + Ka1[:, -1])
    ms['column_out']['time'] = t
    ms['column_out']['CV'] = t / ms['elution_RT']
    ms['column_out']['IS_pos'] = ms['column_out']['cNa'] + ms['column_out']['cTH']
    ms['column_out']['IS_neg'] = ms['column_out']['cCl'] + ms['column_out']['cA_ion']
   
    return ms

    
def titrate(ms, pH, pKa1, CA, pKa2, CT, S):
    # unpack constants
    A1 = ms['wash_buffer']['A']
    A2 = ms['elution_buffer']['A']
    b1 = ms['wash_buffer']['b']
    b2 = ms['elution_buffer']['b']
    zA1 = ms['wash_buffer']['zA']
    zA2 = ms['elution_buffer']['zA']
    zI = ms['zI'] # Net charge on H+

    # Constants
    Kw = 10**-14
    zI = 1  # Net charge on H+

    # Iterate over ionic strength dependent items
    S_initial_salt = S
    for j in range(10):
        g1 = activity_coeff(S, A1, b1, zI)  # activity coeff H+: g(S)
        cH = (10**-pH) / g1                 # 'Target' cH(g(S))
        Ka1 = equilib_constant(S, A1, b1, zA1, pKa1)  # Ka1(S)
        Ka2 = equilib_constant(S, A2, b2, zA2, pKa2)  # Ka2(S)

        conc_titrant = (Ka1 * CA / (cH + Ka1)) + (Kw / cH) - (CT * cH / (cH + Ka2)) - cH

        # Get sum of all charged species after 'titration' to final pH
        # Acetate-          OH-         TrisH+          H+  Na+(or Cl-)
        ionic_conc_at_target_ph = (Ka1 * CA / (cH + Ka1)) + (Kw / cH) + (CT * cH / (cH + Ka2)) + cH + abs(conc_titrant)
        # Another 0.5 multiplier to convert ionic conc to IS (assumed all monovalent ions here)
        S = S_initial_salt + 0.5 * ionic_conc_at_target_ph

    final_IS = S
    return conc_titrant, final_IS

    
def activity_coeff(S, A, b, z):
    g = 10 ** (-((z ** 2) * (A * np.sqrt(S) / (1 + np.sqrt(S)) - b * S)))
    return g

    
def equilib_constant(S, A, b, zA, pKa):
    ka = 10 ** -(pKa + 2 * (zA - 1) * (A * np.sqrt(S) / (1 + np.sqrt(S)) - b * S))
    return ka


def get_cH_quadratic_neutrality(cNa, cCl, Ka1, Ka2, cA, cT):
    # Assumption: DNaCl >> [H+], DNaCl >> [OH-]
    # These should be total Tris and Acetate, not ionic or uncharged species only
    DNaCl = cNa - cCl
    BI = (DNaCl * (Ka1 + Ka2) + Ka2 * (cT - cA)) / (DNaCl + cT)
    CI = (DNaCl - cA) * Ka1 * Ka2 / (DNaCl + cT)
 
    discriminant = BI ** 2 - 4 * CI

    # Get quadratic roots and pick the larger one
    root1 = np.real((-BI + np.sqrt(discriminant)) / 2)
    root2 = np.real((-BI - np.sqrt(discriminant)) / 2)
    
    cH = np.maximum(np.real(root1), np.real(root2))

    return cH


def get_IS_from_cH(cH, Ka1, cA, Ka2, cT, cNa, cCl):
    Kw = 1e-14
    
    ionic_conc = (Ka1 * cA / (cH + Ka1)) + (Kw / cH) + (cT * cH / (cH + Ka2)) + cH + cNa + cCl
    IS = ionic_conc * 0.5
    
    return IS

'''
def dsys(t, y, ms):
    dydt = np.zeros(5*60+5)
    return dydt
'''

def dsys(t, y, ms):
    # Unpack inputs from model structure
    cNa0 = ms['cNa']
    cCl0 = ms['cCl']
    cT0 = ms['cT_total']
    cA0 = ms['cA_total']
    time = [t / 60 for t in ms['inlet_time']] # minutes
    
    # Buffer details   
    pKa1, zA1, A1, b1 = unpack_buffer(ms['wash_buffer'])
    pKa2, zA2, A2, b2 = unpack_buffer(ms['elution_buffer'])
    
    # Resin details
    K = ms['ligand_K']
    qR = ms['ligand_density'] * 1e-3 # M
    Et = ms['Et']
    u =  ms['col_int_vel'][2] * 60 * 100 # Interstitial velocity (cm/min) for elution section
    k_kin = ms['k_kin_titration']
    N = ms['N']
    h = ms['col_length'] * 100 / N # cm
    zI = ms['zI']

    # Unpack y vector
    cT = y[0*N:1*N+1]
    cA = y[1*N+1:2*N+2]
    cCl = y[2*N+2:3*N+3]
    cNa = y[3*N+3:4*N+4]
    qNa = y[4*N+4:5*N+5]

    # Zeros dot vectors
    dydt = np.zeros(5*N+5)
    dcTdt = np.zeros(N+1)
    dcAdt = np.zeros(N+1)
    dcCldt = np.zeros(N+1)
    dcNadt = np.zeros(N+1)
    dqNadt = np.zeros(N+1)
    
    cH = np.zeros(N+1)
    IS = np.zeros(N+1)
    Ka1 = np.zeros(N+1)
    Ka2 = np.zeros(N+1)
    
    # Find the index i where the given time point falls
    i = 0
    for j in range(len(time)):
        if time[j] >= t:
            i = j-1
            break
 
    # update initial condition by linear interpolation of the inlet profile
    mult = (t - time[i]) / (time[i+1] - time[i])

    cNa[0] = cNa0[i] + (cNa0[i+1] - cNa0[i]) * mult
    cCl[0] = cCl0[i] + (cCl0[i+1] - cCl0[i]) * mult
    cT[0] = cT0[i] + (cT0[i+1] - cT0[i]) * mult
    cA[0] = cA0[i] + (cA0[i+1] - cA0[i]) * mult

    # Boundary conditions at inlet, time variant
    IS[0] = max(cNa[0], cCl[0])  # Initialize IS
    DeltaIS = 1
    
    # Iterate over IS and pH to get final IS and pH values
    while abs(DeltaIS) > 0.0001:
        # Ionic strength dependent terms
        Ka1[0] = equilib_constant(IS[0], A1, b1, zA1, pKa1)
        Ka2[0] = equilib_constant(IS[0], A2, b2, zA2, pKa2)
  
        # Quadratic electroneutrality equation to get H+ concentration
        cH[0] = get_cH_quadratic_neutrality(cNa[0], cCl[0], Ka1[0], Ka2[0], cA[0], cT[0])
        
        # Given cH get IS
        IS_1_new = get_IS_from_cH(cH[0], Ka1[0], cA[0], Ka2[0], cT[0], cNa[0], cCl[0])
        DeltaIS = IS[0] - IS_1_new  # Store the difference 
        IS[0] = IS_1_new            # Update ionic strength

    # -------------------------------------------------------------------------
    # Set up differential balances for all discretization cells
    for i in range(1, N+1):
        # Transport of non-interacting components by convection (mol / L / min)
        # !!! changed to use interstitial velocity calculated using Ee rather than Et
        dcTdt[i] = u * (cT[i-1] - cT[i]) / h
        dcAdt[i] = u * (cA[i-1] - cA[i]) / h
        dcCldt[i] = u * (cCl[i-1] - cCl[i]) / h

        # Ionic strength dependent terms, including activity coefficient
        IS[i] = IS[i-1]
        Ka1[i] = equilib_constant(IS[i], A1, b1, zA1, pKa1)
        Ka2[i] = equilib_constant(IS[i], A2, b2, zA2, pKa2)
        
        # Quadratic electroneutrality equation to get H+ concentration     
        cH[i] = get_cH_quadratic_neutrality(cNa[i], cCl[i], Ka1[i], Ka2[i], cA[i], cT[i]) 
        IS[i] = get_IS_from_cH(cH[i], Ka1[i], cA[i], Ka2[i], cT[i], cNa[i], cCl[i])

        #### Transport of Na+ ions, solid and liquid phase
        # equilibrium Na+ concentration (where does this come from?)
        # !!! should multiply qR by zI?
        cNa_eq = qNa[i]**2 * cH[i] / ((qR - qNa[i]) * K)
        # adsorption of sodium ions - linear driving force
        dqNadt[i] = k_kin * (cNa[i] - cNa_eq)
        # sodium ions, convective transport minus solid phase ratio times
        # rate of adsorption of sodium ions
        dcNadt[i] = (u * (cNa[i-1] - cNa[i]) / h) - (((1 - Et) / Et) * dqNadt[i])
                
    # -------------------------------------------------------------------------
    # print(t)
    
    # Transfer back to ydot vector for solving
    dydt[0*N:1*N+1] = dcTdt
    dydt[1*N+1:2*N+2] = dcAdt
    dydt[2*N+2:3*N+3] = dcCldt
    dydt[3*N+3:4*N+4] = dcNadt
    dydt[4*N+4:5*N+5] = dqNadt
    
    return dydt


#### Plotting and Saving Functions ############################################

def make_excel(ms, path):    
    file = os.path.join(path, 'resin_titration_out.xlsx')
    with pd.ExcelWriter(file) as writer:       
        df = pd.DataFrame(ms['column_out'])
        df.to_excel(writer, sheet_name='column_out', header = True, index = False)
        
        
def plot_titration(ms, path):
    CV = ms['column_out']['CV']
    pH_in = ms['pH_in']
    pH_out = ms['column_out']['pH']
    font = 16
    figname = os.path.join(path, 'titration_plot.png')
    
    # Create a figure and a grid of subplots
    fig, ax1 = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    
    #### component plot ######################################################
    # First plot with two y-axes
    ax_pH1 = ax1[1] # pH axis on component plot
    ax_c = ax_pH1.twinx() # concentration axis on component plot
    ax_pH2 = ax1[0]

    # plot pH
    l1, = ax_pH1.plot(CV, pH_out, 'k-', linewidth=2, label='pH')
    
    # plot ion species
    l2, = ax_c.plot(CV, ms['column_out']['cNa'], '-', color=[0.816, 0, 0.086, 1], linewidth=1.5, label='Na+ cation')
    l3, = ax_c.plot(CV, ms['column_out']['cCl'], '-', color=[0.192, 0.196, 0.231, 1], linewidth=1.5, label='Cl- anion')
    l4, = ax_c.plot(CV, ms['column_out']['cA'], '-', color=[0.192, 0.345, 0.612, 1], linewidth=1.5, label='Acetic acid')
    l5, = ax_c.plot(CV, ms['column_out']['cA_ion'], '-', color=[0.725, 0.716, 0.678, 1], linewidth=1.5, label='Acetate anion')
    l6, = ax_c.plot(CV, ms['column_out']['cT'], '-', color=[0.329, 0.188, 0.502, 1], linewidth=1.5, label='Tris')
    l7, = ax_c.plot(CV, ms['column_out']['cTH'], '-', color=[0.659, 0.604, 0.098, 1], linewidth=1.5, label='TrisH+ cation')
  
    lines = [l1, l2, l3, l4, l5, l6, l7]
    
    ax_pH1.set_ylim([0, 10])
    ax_pH1.set_xlim(0, CV[-1])
    ax_pH1.set_ylabel('pH', fontsize=font)
        
    # combine legends for both axes into one
    labels = [line.get_label() for line in lines]
    legend = ax_pH1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.75, 1.1), frameon=True, ncol=1, fontsize=12)
    legend.get_frame().set_alpha(0)  # Adjust the value between 0 and 1
    plt.subplots_adjust(right=0.5)

    ax_c.set_ylabel('(M)', fontsize=font)
    ax_c.set_ylim([0, 0.4])
    ax_pH1.set_xlabel('CV', fontsize=font)
    ax_pH1.tick_params(axis='both', which='both', direction='in', color='k')
    ax_c.tick_params(axis='both', which='both', direction='in', color='k')
    
    # Add textbox run information
    textbox_props = dict(edgecolor='none', facecolor='none')
    plt.gca().annotate('\n'.join([
        'Wash buffer:',
        f'Tris: {ms["wash_buffer"]["buffer1_total"]} mM',
        f'Acetate: {ms["wash_buffer"]["buffer2_total"]} mM',
        f'NaCl: {ms["wash_buffer"]["NaCl"]} mM\n',
        'Elution buffer:',
        f'Tris: {ms["elution_buffer"]["buffer1_total"]} mM',
        f'Acetate: {ms["elution_buffer"]["buffer2_total"]} mM',
        f'NaCl: {ms["elution_buffer"]["NaCl"]} mM',
    ]), xy=(1.3, -0.05), xycoords='axes fraction', bbox=textbox_props, fontsize=12)

    #### pH plot ######################################################
    ax_pH2.set_ylim([0, 10])
    ax_pH2.set_xlim(0, CV[-1])
    ax_pH2.set_ylabel('pH', fontsize=font)

    ax_pH2.set_xlabel('CV', fontsize=font)
    ax_pH2.tick_params(axis='both', which='both', direction='in', color='k')
        
    ax_pH2.plot(CV, pH_in, 'k-', linewidth=2, label='pH at column inlet')
    ax_pH2.plot(CV, pH_out, 'r-', linewidth=2, label='pH at column outlet')
    
    ax_pH2.legend(loc='upper right', frameon=True, ncol=1, fontsize=12)
    
    ###########################################################################
    # Save the figure
    fig.tight_layout()
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()
                                  