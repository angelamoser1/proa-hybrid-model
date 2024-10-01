import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import os
import pandas as pd
from addict import Dict


def find_closest_element(arr, target_value):
    closest_index = np.abs(arr - target_value).argmin()
    closest_element = arr[closest_index]
    return closest_index, closest_element


class InletProfile:
    """
    A class that contains the %B signal supplied to the pumps
    """
    def __init__(self, time_bounds, section_times):
        
        # Global times define total time the pumps are run
        self.global_start_time = time_bounds[0]
        self.global_end_time = time_bounds[1]
        self.number_of_steps = time_bounds[2]
        # Section times define section boundaries
        self.section_times = section_times
        # Number of sections = number of section boundaries + 1
        self.nsections = len(self.section_times)+1
        
        # Construct the linspace for the total time
        self.time_series = np.linspace(self.global_start_time,
                                       self.global_end_time, 
                                       self.number_of_steps)
        
        # Define inlet polynomial terms for each section 
        self.section_constant = np.zeros(self.nsections)        # mol/(m^3)
        self.section_linear = np.zeros(self.nsections)          # mol/(m^3.s)
        
    def construct_inlet_profile(self):
        """
        Construct the %B signal supplied to the pumps based on each section
        """
        # Get the section boundaries for the constructed linspace (we need to 
        # do this since the user-supplied points may not exist in the linspace)
        closest_section_times = np.zeros(len(self.section_times))
        closest_section_times_index = np.zeros(len(self.section_times))
        for i, time in enumerate(self.section_times):
            closest_section_times_index[i], closest_section_times[i] = find_closest_element(
                self.time_series, time)
        
        # Construct the actual inlet signal, based on the section boundaries
        # obtained from the linspace
        self.inlet_profile = np.zeros(len(self.time_series))        
        # Section counter to update inlet polynomial terms
        section_counter=0
        
        # Loop over each timepoint in the global time series
        for i,time_i in enumerate(self.time_series):
            # If timepoint is in first section
            if section_counter == 0:
                section_lower_index = 0                                                           # Section start index
                section_upper_index = int(closest_section_times_index[section_counter])           # Section end index
                section_start_time = self.time_series[section_lower_index]                        # Section start tune
                a = self.section_constant[section_counter]
                b = self.section_linear[section_counter]
                
            # Else if timepoint is in an intermediate sections
            elif 0 < section_counter < self.nsections-1 :
                section_lower_index = int(closest_section_times_index[section_counter-1])
                section_upper_index = int(closest_section_times_index[section_counter])
                section_start_time = self.time_series[section_lower_index+1]
                a = self.section_constant[section_counter]
                b = self.section_linear[section_counter]
                
            # Else if time point is in the last section
            elif section_counter == self.nsections-1:
                section_lower_index = int(closest_section_times_index[section_counter-1])
                section_upper_index = len(self.time_series) - 1
                section_start_time = self.time_series[section_lower_index]
                a = self.section_constant[section_counter]
                b = self.section_linear[section_counter]
                
            # Increment section counter
            if i > section_upper_index:
                section_counter = section_counter + 1
                
            # Calculate inlet signal y value for each time point
            self.inlet_profile[i] = a + b*(time_i-section_start_time)
                    
        return self.time_series, self.inlet_profile

class Buffer:
    def __init__(self, name):
        self.name = name
        
        # Create list of buffer component concentrations
        self.Na             = 0
        self.Cl             = 0
        self.Acetate_total  = 0
        self.Tris_total     = 0
        self.IS             = 0.5*(self.Na + self.Cl)
        self.guess_pH       = 10
        self.guess_H        = 10**(-self.guess_pH)
        
    def calculate_IS(self):
        self.IS             = 0.5*(self.Na + self.Cl)
        
class Mixer:
    def __init__(self, V, Q, CA0, inlet_time, inlet_data):
       self.volume          = V
       self.flow            = Q
       self.init_conc       = CA0
       self.inlet_time      = inlet_time
       self.inlet_data      = inlet_data
       
    def solve_cstr_model(self):
        y0 = [self.init_conc] # Initial concentration in the mixer
        
        # Solve mixing ODEs using odeint - get CSTR outlet function
        cstr_output = odeint(cstr_model, y0, self.inlet_time,
                             args=(self.inlet_time, self.inlet_data, 
                                   self.flow, self.volume))
        self.output_function = cstr_output[:,0]
        return self.output_function
        
    def mix_buffers(self, BufferA, BufferB):
        """
        Mix initial buffer A with end-point buffer B using mixer output function
        """
        frac_b = self.output_function
        frac_a = 1-self.output_function
        frac_a[frac_a < 1e-12] = 0

        conc_Na         = BufferB.Na*frac_b             + BufferA.Na*frac_a
        conc_Cl         = BufferB.Cl*frac_b             + BufferA.Cl*frac_a
        conc_Acetate    = BufferB.Acetate_total*frac_b  + BufferA.Acetate_total*frac_a
        conc_Tris       = BufferB.Tris_total*frac_b     + BufferA.Tris_total*frac_a
        conc_guess_H    = BufferB.guess_H*frac_b        + BufferA.guess_H*frac_a
        
        component_array = [conc_Acetate, conc_Tris, conc_Na, conc_Cl, conc_guess_H]
        return component_array
        
    def calculate_pH_values(self, component_array, IS_dependence=False):
        # Unpack component arrays
        Ac_t_array      = component_array[0]
        Tris_t_array    = component_array[1]
        Na_array        = component_array[2]
        Cl_array        = component_array[3]
        H_array         = component_array[4]
        
        # Define constants of the system
        # A units (sqrt(kg)/sqrt(mol)) - needs to inverse of sqrt(ionic strength)
        # assuming density of 1 we can use mol/L
        # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(LibreTexts)/25%3A_Solutions_II_-_Nonvolatile_Solutes/25.07%3A_Extending_Debye-Huckel_Theory_to_Higher_Concentrations
        Kw          = 1e-14
        pKa_acetate = 4.76
        pKa_tris    = 8.07
        Ka_acetate  = 10 ** (-pKa_acetate)
        Ka_tris     = 10 ** (-pKa_tris)
        A           = 0.51 # Debye-Huckel theory constant 1.825*(10**6)*eps*(T**1.5) at 25 C water
        B           = 0.33 # B
        
        # Buffer equilibration equations: # solution is [H+, Acetate_ion, Tris]
        def fun(x, Ac_t, Tris_t):    
            Ac_i = Ac_t     * Ka_acetate / (Ka_acetate + x[0])      # Acetate ion
            Tris = Tris_t   * Ka_tris / (Ka_tris + x[0])            # Tris (uncharged)
            return x[0] + Na + (Tris_t-x[2]) - (Kw / x[0]) - Cl - x[1], Ac_i - x[1], Tris - x[2]
        
        # Equilibrate the compositions at each time point
        pH_values = []
        IS_values = []
        Acetate_ion = []
        Acetic_acid = []
        TrisH_ion = []
        Tris = []
        for i in range(len(Ac_t_array)):
            # Get composition values at a given time point
            Ac_t    = Ac_t_array[i]
            Tris_t  = Tris_t_array[i]
            Na      = Na_array[i]
            Cl      = Cl_array[i]
            if i == 0: # Guess concentrations for first time point
                H = H_array[i]
                IS = 0.5*(Na_array[i]+Cl_array[i])
            else:   # Dynamically update guess H+ values at each time point
                H = 10**(-pH_values[i-1])
                if IS_dependence == True:
                    pKa_acetate = 4.76 + (0.51*np.sqrt(IS))/(1+1.332*np.sqrt(IS))
                    pKa_tris    = 8.07 - (0.51*np.sqrt(IS))/(1+1.332*np.sqrt(IS))
                    Ka_acetate  = 10 ** (-pKa_acetate)
                    Ka_tris     = 10 ** (-pKa_tris)
                    print(pKa_tris)
                
            # Construct overall initial guess list
            x0 = [H, Ac_t, Tris_t]
            
            # Solve for each time-point to get equilibrated pH and IS vs time
            solution = fsolve(fun, x0, args=(Ac_t,Tris_t,))
            IS = 0.5*(solution[1] + (Tris_t-solution[2]) + Na + Cl + solution[0] + Kw/solution[0])
            
            # Store pH and IS
            solution[0] = -np.log10(solution[0])
            pH_values.append(solution[0])
            IS_values.append(IS)
            Acetate_ion.append(solution[1])
            Acetic_acid.append(Ac_t - solution[1])
            Tris.append(solution[2])
            TrisH_ion.append(Tris_t-solution[2])
        
        # Set values less than 0.001 to 0
        Acetate_ion = [x if x >= 1e-12 else 0 for x in Acetate_ion]
        Acetic_acid = [x if x >= 1e-12 else 0 for x in Acetic_acid]
        TrisH_ion = [x if x >= 1e-12 else 0 for x in TrisH_ion]
        Tris = [x if x >= 1e-12 else 0 for x in Tris]
        
        cstr_out = {
            'pH': pH_values,
            'ionic_strength': IS_values,
            'Acetate_ion': Acetate_ion,
            'Acetic_acid': Acetic_acid,
            'Tris': Tris,
            'TrisH_ion': TrisH_ion
            }
        
        return cstr_out
    
# Define the CSTR model as a system of ODEs; Only buffer mixing occurs here
def cstr_model(y, t, time, CA_inlet_data, Q, V):
    # Supply CA_inlet as an interpolated function of the time-series inlet function
    CA_inlet = np.interp(t, CA_inlet_data, CA_inlet_data)
    CA = y
    dCA_dt = (Q / V) * (CA_inlet - CA)
    return dCA_dt


def get_CSTR_profile(ms):
    ms['step_time'] = ms['elution_CV'] * ms['elution_RT'] * 60 # [sec]
    # Create inlet profile object and the inlet function to supply to the mixer
    input_signal                    = InletProfile([0, ms['step_time'], 1000], [0])
    input_signal.section_constant   = np.array([0, 1])
    input_signal.section_linear     = np.array([0, 0])
    inlet_time, inlet_function      = input_signal.construct_inlet_profile()
    time_series_inlet_function      = [inlet_time, inlet_function] # Supply to CSTR
    
    # Model the gradient mixer and get the CSTR-transformed A/B buffer fractions
    V       = ms['mixer_volume'] # Volume of the CSTR (m^3)
    Q       = ms['flow_rate'][2] # use elution step flow rate [m^3/s]
    CA0     = 0                # Initial concentration in tank, normalized units
    CSTR = Mixer(V, Q, CA0, inlet_time, inlet_function)
    output_function = CSTR.solve_cstr_model()

    # Define initial buffer - A (all in M)
    buffer_a = Buffer("Wash Buffer")
    buffer_a.Na = ms['wash_buffer']['Na'] * 1e-3
    buffer_a.Cl = ms['wash_buffer']['Cl'] * 1e-3
    buffer_a.Tris_total = ms['wash_buffer']['buffer1_total'] * 1e-3
    buffer_a.Acetate_total = ms['wash_buffer']['buffer2_total'] * 1e-3
    
    # Define final buffer - B (all in M)
    buffer_b = Buffer("Elution Buffer")
    buffer_b.Na = ms['elution_buffer']['Na'] * 1e-3
    buffer_b.Cl = ms['elution_buffer']['Cl'] * 1e-3
    buffer_b.Tris_total = ms['elution_buffer']['buffer1_total'] * 1e-3
    buffer_b.Acetate_total  = ms['elution_buffer']['buffer2_total'] * 1e-3

    # Buffer gradient: A - B1
    component_array = CSTR.mix_buffers(buffer_a, buffer_b)
    # print(component_array)
    # Get outlet equilibrated pH and ionic strength
    cstr_out = CSTR.calculate_pH_values(component_array, IS_dependence=False)

    ms['inlet_time'] = inlet_time
    ms['inlet_function'] = inlet_function 
    ms['output_function'] = output_function
    ms['component_array'] = component_array
    ms['cstr_out'] = cstr_out

    return ms


def make_excel(ms, path):    
    file = os.path.join(path, 'cstr_out.xlsx')
    with pd.ExcelWriter(file) as writer:       
        df = pd.DataFrame(ms['cstr_out'])
        df.to_excel(writer, sheet_name='CSTR_out', header = True, index = False)
    
    
def plot_CSTR(ms, path):
    # unpack stuff
    inlet_CVs = ms['inlet_time'] / (ms['elution_RT']*60)
    step_CVs = ms['elution_CV']
    inlet_function = ms['inlet_function']
    output_function = ms['output_function']
    component_array = ms['component_array']
    cstr_out = ms['cstr_out']
    font_size = 20
    RPI_cherry = [0.816, 0, 0.086, 1]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot the concentration of A over time
    ax1.plot(inlet_CVs, inlet_function, label='no CSTR', color=[0.192, 0.345, 0.612, 1])
    ax1.plot(inlet_CVs, output_function, label='CSTR online', color=RPI_cherry)
    ax1.set_xlim(0, step_CVs)
    ax1.set_xlabel('CVs', fontsize=font_size)
    ax1.set_ylabel('%B ', fontsize=font_size)
    ax1.set_title('CSTR Mixing Simulation: Buffer fraction', fontsize=font_size)
    ax1.legend(fontsize=15)
    ax1.tick_params(axis='both', which='major', direction='in', length=10, labelsize=font_size)
    
    # Plot buffer components over time
    ax2.plot(inlet_CVs,    ms['cstr_out']['Acetate_ion'], color=[0.725, 0.716, 0.678, 1], label='Acetate anion')
    ax2.plot(inlet_CVs,    ms['cstr_out']['Acetic_acid'], color=[0.192, 0.345, 0.612, 1], label='Acetic acid')
    ax2.plot(inlet_CVs,    ms['cstr_out']['Tris'], color=[0.329, 0.188, 0.502, 1], label='Tris')
    ax2.plot(inlet_CVs,    ms['cstr_out']['TrisH_ion'], color=[0.659, 0.604, 0.098, 1], label='TrisH+ cation')
    ax2.plot(inlet_CVs,    component_array[2], color=RPI_cherry, label='Na+ cation')
    ax2.plot(inlet_CVs,    component_array[3], color=[0.192, 0.196, 0.231, 1] , label='Cl- anion')
    ax2.set_xlim(0, step_CVs)
    ax2.set_xlabel('CVs', fontsize=font_size)
    ax2.set_ylabel('Concentration, M', fontsize=font_size)
    ax2.set_title('CSTR Mixing Simulation:\nComponent concentrations', fontsize=font_size)
    ax2.legend(fontsize=15)
    ax2.tick_params(axis='both', which='major', direction='in', length=10, labelsize=font_size)
    
    # Plot IS over time
    ax3.plot(inlet_CVs, ms['cstr_out']['ionic_strength'],  color=RPI_cherry)
    ax3.set_xlim(0, step_CVs)
    ax3.set_xlabel('CVs', fontsize=font_size)
    ax3.set_ylabel('Ionic strength, M', fontsize=font_size)
    ax3.set_title('CSTR Mixing Simulation: Ionic strength', fontsize=font_size)
    ax3.tick_params(axis='both', which='major', direction='in', length=10, labelsize=font_size)
    
    # Plot pH over time
    ax4.plot(inlet_CVs, ms['cstr_out']['pH'],  color=RPI_cherry)
    ax4.set_xlim(0, step_CVs)
    ax4.set_xlabel('CVs', fontsize=font_size)
    ax4.set_ylabel('pH', fontsize=font_size)
    ax4.set_title('CSTR Mixing Simulation: pH', fontsize=font_size)
    ax4.tick_params(axis='both', which='major', direction='in', length=10, labelsize=font_size)
    
    fig.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(path, 'CSTR_mixing_sim_plots.png'), dpi=300) 
    plt.show()
    

    

    
