# -*- coding: utf-8 -*-
"""
This script contains the functions used for transport parameter calculations
for the CADET chromatography model. It takes the model structure as input
assuming all inputs are already in the correct (SI) units and adds the
calculated transport parameters to the model structure.
"""
import numpy as np

def calculate_transport(ms, T=237, viscosity=0.001, density=1000):
    # Temperature [Kelvin], default is 297 for ~24 Celsius room temperature
    # Viscosity [kg/(m*s)], default 0.001 for water          
    # Density [kg/m^3],  default 1000 for water
                     
    # unpack variables from ms
    load_tubing_id = ms['load_tubing_id']
    elu_tubing_id = ms['elu_tubing_id']
    particle_diameter = ms['particle_diameter']
    Ee = ms['Ee'] 
    Ep = ms['Ep'] 
    flow_rate = ms['flow_rate']
    col_length = ms['col_length']
    col_lin_vel = ms['col_lin_vel']
    MW = ms['MW']
    
    #### Transport parameter calculations ######################################### 
   
    # component dependent parameters
    r_hyd = np.zeros(len(MW))
    Do = np.zeros(len(MW))
    Sc = np.zeros(len(MW))
    for i in range(len(MW)):
        # calculate hydrodynamic radius        
        r_hyd[i] = hydrodynamic_radius(MW[i])

        # Calculate molecular diffusivities [m^2/s],    
        Do[i] = molecular_diffusivity(r_hyd[i], viscosity, T)
        
        # Calculate one Scanlon number for each component
        Sc[i] = scanlon_number(Do[i], viscosity, density)
        
    # section dependent parameters
    Re = np.zeros(len(col_lin_vel))
    for i in range(len(col_lin_vel)):
        # Calculate one Reynolds number for each section
        Re[i] = reynolds_number(col_lin_vel[i], particle_diameter, viscosity, density)
    
    # component and section dependent parameters
    Sh = np.zeros((len(MW), len(col_lin_vel)))
    k_film = np.zeros((len(MW), len(col_lin_vel)))
    Pe = np.zeros((len(MW), len(col_lin_vel)))
    Dax_col = np.zeros((len(MW), len(col_lin_vel))) 
    load_tubing_Dax = np.zeros((len(MW), len(col_lin_vel)))
    elu_tubing_Dax = np.zeros((len(MW), len(col_lin_vel)))
    for i in range(len(MW)):
        for j in range(len(col_lin_vel)):
            # Calculate one Sherwood number for each component in each section
            Sh[i][j] = sherwood_number(Sc[i], Re[j], Ee)
    
            # Calculate one film mass transfer coefficient for each component in each section
            k_film[i][j] = film_mass_transfer_coefficient(Do[i], Sh[i][j],
                                    particle_diameter)
            
            # Calculate one Peclet number for each component in each section
            Pe[i][j] = peclet_number(Do[i], col_length, col_lin_vel[j], particle_diameter,
                                Ee, Re[j])
             
            # Column axial dispersion [m^2/s]
            Dax_col[i][j] = column_axial_dispersion(col_lin_vel[j], col_length, Pe[i][j])

            # load tubing axial dispersion [m^2/s]
            load_tubing_Dax[i][j] = tubing_Dax(load_tubing_id, Do[i], flow_rate[j])
            
            # elution tubing axial dispersion [m^2/s]
            elu_tubing_Dax[i][j] = tubing_Dax(elu_tubing_id, Do[i], flow_rate[j])


    # Effective pore diffusivity for modifier from Mackie-Meares correlation
    modifier_pore_diff = mackie_mears(Do[0], Ep)
        
    # update ms with calculated transport parameters
    ms['r_hyd'] = r_hyd
    ms['k_film'] = k_film 
    ms['Dax_col'] = Dax_col
    ms['load_tubing_Dax'] = load_tubing_Dax 
    ms['elu_tubing_Dax'] = elu_tubing_Dax
    ms['modifier_pore_diff'] = modifier_pore_diff
    
    if ms.get('estimate_pore_diffusivity') == True:
        r_pore = ms['r_pore']
        Ep_prot = ms['Ep_prot']
        pore_diff = [estimate_pore_diffusivity(Ep_prot, r_pore, r, d)
                     for r, d in zip(r_hyd[1:], Do[1:])]
        ms['pore_diff'] = pore_diff
      
    return ms


def hydrodynamic_radius(MW):
    # Calculate hydrodynamic radius from globular ptotein correlation [nm]
    r_hyd = 1e-9*0.7429*MW**0.3599
    return r_hyd


def molecular_diffusivity(r_hyd, viscosity, T):  
    # Calculate molecular diffusivity [m^2/s]
    # Boltzmann constant [m^2*kg/(s^2*K)]
    Kb = 1.38064e-23                       
    Do = Kb*T/(6*np.pi*viscosity*r_hyd)
    return Do


def reynolds_number(linear_velocity, particle_diameter, viscosity, density):
    Re = density*linear_velocity*particle_diameter/viscosity
    return Re


def scanlon_number(molecular_diffusivity, viscosity, density):
    Sc = viscosity/(density*molecular_diffusivity) 
    return Sc


def peclet_number(molecular_diffusivity, col_length, col_lin_vel, particle_diameter, Ee, Re):    
    # Column peclet number, dispersion - Rastegar & Gu 2017
    Pe = (0.7*molecular_diffusivity/(col_length*col_lin_vel)
          + (particle_diameter/col_length)
          * (Ee/(0.18 + 0.008*Re**0.59)))**-1
    return Pe


def film_mass_transfer_coefficient(molecular_diffusivity, Sh, particle_diameter):
    # Film mass transfer coeff. [m/s]
        k_film = Sh*molecular_diffusivity/particle_diameter 
        return k_film


def sherwood_number(Sc, Re, Ee):
    # Sherwood number, mass transfer - Wilson & Geankopolis 1966
    Sh = (1.09*Re**0.33*Sc**0.33)/Ee
    return Sh


def column_axial_dispersion(col_lin_vel, col_length, Pe):
    # Column axial dispersion [m^2/s]
    Dax_col = col_lin_vel*col_length/Pe
    return Dax_col
        

def tubing_Dax(tubing_id, Do, flow_rate):
    # Dispersion for tubing - Taylor expression (1953)
    tubing_lin_vel = flow_rate / (0.25*np.pi*tubing_id**2)
    # Bond number
    Bo = tubing_lin_vel*tubing_id/Do
    if Bo > 100:
        Dax = tubing_lin_vel**2*tubing_id**2/(192*Do)
    else:
        Dax = Do + tubing_lin_vel**2*tubing_id**2/(192*Do)
    return Dax


def mackie_mears(molecular_diffusivity, Ep):
    # Effective pore diffusivity for modifier, Mackie-Meares correlation
    Dp = (Ep/(2-Ep))**2*molecular_diffusivity
    return Dp


# function to estimate pore diffusivity from an empirical correlation
def estimate_pore_diffusivity(Ep_prot, r_pore, r_hyd, Do):        
    # Pore diffusivity for protein based on tortuosity estimate
    tortuosity = 4 # literally a guess but what can you do ¯\_(ツ)_/¯
    lambda_m = r_hyd / r_pore
    
    # diffusional hinderance coefficient from Jungbauer and Carta
    # diff_hind = 1 + 1.125 * lambda_m * math.log(lambda_m) - 1.539 * lambda_m
    
    # diffusional hinderance coefficient correlation from Dechadilok and Deen 2006
    diff_hind = (1 + 1.125 * lambda_m * np.log(lambda_m) 
                - 1.5604 * lambda_m 
                + 0.528155 * lambda_m**2
                + 1.91521 * lambda_m**3
                - 2.81903 * lambda_m**4
                + 0.270788 * lambda_m**5
                + 1.10115 * lambda_m**6
                - 0.435933 * lambda_m**7)
    
    pore_diff = Ep_prot * diff_hind * Do / tortuosity
    
    return pore_diff