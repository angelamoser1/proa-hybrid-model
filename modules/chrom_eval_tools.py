'''
This script contains basic functions for analyzing chromatograms.
'''

import numpy as np 
from os.path import join

    
def select_data(path, name, xcol, ycol):
    file = join(path, name)
    skip_header = 2
    try:
        xdata = np.genfromtxt(file, skip_header=skip_header, usecols=(int(xcol)), delimiter=',')
        ydata = np.genfromtxt(file, skip_header=skip_header, usecols=(int(ycol)), delimiter=',')
    except:
        try:
            xdata = np.genfromtxt(file, skip_header=skip_header, usecols=(int(xcol)), delimiter='\t')
            ydata = np.genfromtxt(file, skip_header=skip_header, usecols=(int(ycol)), delimiter='\t')
        except:
            try:
                xdata = np.genfromtxt(file, skip_header=skip_header, usecols=(int(xcol)), delimiter='\t', encoding='utf-16')
                ydata = np.genfromtxt(file, skip_header=skip_header, usecols=(int(ycol)), delimiter='\t', encoding='utf-16')
            except:
                message = f'Could not select data from file {name}. Try saving your data as a CSV UTF-8 (comma delimited) (*.csv) file using excel.'
                return 1, message
    xdata = xdata[~np.isnan(xdata)]
    ydata = ydata[~np.isnan(ydata)]
    return xdata, ydata
    
  
# input only y data, finds index of peak max and peak start and end given
# a threshold
def detect_peak(data, start_threshold, end_threshold):
    index_max = np.argmax(data)
    if max(data) < start_threshold:
        # if peak threshold is not reached, return 0 for start and end
        return index_max, 0, 0
    else:
        x = index_max
        index_start = 0
        index_end = 0
        for point in data[:index_max]:
            if data[x] >= start_threshold and data[x-1] < start_threshold:
                index_start = x
                break
            else: x -= 1
        y = index_max
        for point in data[index_max:]:
            if data[y] <= end_threshold and data[y-1] > end_threshold:
                index_end = y
                break
            else: y += 1
        return index_max, index_start, index_end


# determine the baseline using a region you specify and subtract from y
def subtract_baseline(xdata, ydata, baseline_section):
    # select the data in the region you would like to use for the baseline
    baseline_data = ydata[np.where(((xdata > baseline_section[0]) & (xdata < baseline_section[1])))]
    # Calculate the baseline as the average in the selected region
    baseline = np.average(baseline_data)
    # subtract from y data
    ydata = ydata - baseline
    return ydata


def get_first_moment(xdata, ydata, idx_low, idx_high):
    # select only the data within the peak
    ypeak = ydata[idx_low:idx_high]
    xpeak = xdata[idx_low:idx_high]
    # Calculate the first moment (centroid)
    numerator = np.sum(xpeak * ypeak)
    denominator = np.sum(ypeak)    
    first_moment = numerator / denominator
    return first_moment

      
# input 2 dfs of the same length
# assumes autozeroed UV before pulse
def evaluate_peak(xdata, ydata):
    # find peak volume with 100 mAu as UV absorbance threshold
    low, high = detect_peak(ydata, 100, 100)
    peak_volume = xdata[high] - xdata[low]
    # baseline = np.mean(ydata[low/2:(low/10)*6])
    baseline = 0 # assumes autozeroed UV before pulse, if not, uncomment above
    ymax = ydata.max()
    peak_height = ymax - baseline
    height10 = peak_height*0.1 + baseline
    height50 = peak_height*0.5 + baseline
    index_max = np.argmax(ydata)
    xmax = xdata[index_max]
    # find peak width at half max height
    low50, high50 = detect_peak(ydata, height50, height50)
    width = xdata[high50]-xdata[low50]
    #find peak assymetry using 10% of max absorbance values
    low10idx, high10idx = detect_peak(ydata, height10, height10)
    low10, high10 = xdata[low10idx], xdata[high10idx]
    A = index_max - low10idx
    B = high10idx - index_max
    asymmetry = B/A
    return ymax, xmax, peak_volume, width, asymmetry, low10, high10


def calculate_DBC(vol_data, c_data, feed_c, CV, end_breakthrough):
    # gets DBC at 10% of feed absorbance
    # assumes baseline already adjusted
    # assumes loading starts at 0 volume and HUV has already been subtracted
    # take only the data up to the end of the breakthrough
    end_idx = (np.abs(vol_data-end_breakthrough)).argmin()
    c_data = c_data[:end_idx]
    # check if 10% of feed concentration has been achieved
    if max(c_data) < 0.10*feed_c:
        DBC10_v = 0
        DBC10 = 0
    else:
        DBC10_idx = (np.abs(c_data-0.10*feed_c)).argmin()
        DBC10_v = vol_data[DBC10_idx]
        DBC10 = DBC10_v*feed_c/CV 
    return DBC10_v, DBC10


def calculate_EBC(vol_data, c_data, feed_c, CV, end_breakthrough):
    # gets EBC at 99% of feed concentration
    # assumes baseline already adjusted
    # assumes loading starts at 0 volume and HUV has NOT been subtracted
    # take only the data from the zero mark up to the end of the breakthrough
    start_idx = (np.abs(vol_data)).argmin()
    end_idx = (np.abs(vol_data-end_breakthrough)).argmin()
    vol_data = vol_data[start_idx:end_idx]
    c_data = c_data[start_idx:end_idx]
    # set threshold for 'complete' breakthrough as a fraction of feed conc
    threshold = 0.99*feed_c
    if max(c_data) < threshold:
        EBC_v = 0
        EBC = 0
    else:
        # loop over data until threshold is reached
        EBC_idx = 0
        for y in range(1,len(c_data)):
            if c_data[y] >= threshold and c_data[y-1] < threshold:
                EBC_idx = y
                break
            else:
                continue
        
        # select from zero to where the threshold is met
        EBC_v_data = vol_data[:EBC_idx]
        EBC_c_data = c_data[:EBC_idx]
        EBC_v = vol_data[EBC_idx]
        # integrate between the feed concentration and the breakthrough curve
        EBC = np.trapz(feed_c-EBC_c_data, EBC_v_data)/CV
    return EBC_v, EBC


# ydata is n_comp x n_points
def evaluate_elution(xdata, ydata, start_threshold, end_threshold):
    # Sum the concentrations of the protein components
    sum_y = np.sum(ydata, axis=0)

    # find peak volume at threshold
    idx_max, idx_start, idx_end = detect_peak(sum_y, start_threshold, end_threshold)
    pool_volume = xdata[idx_end] - xdata[idx_start]

    # calculate the concentration of each component in the pool
    if pool_volume != 0:
        pool_conc = []
        for idx, comp_feed_c in enumerate(ydata):
            pool_mass = np.trapz(ydata[idx][idx_start:idx_end], xdata[idx_start:idx_end])
            # Calculate elution pool concentration
            pool_conc_comp = pool_mass/pool_volume
            pool_conc.append(pool_conc_comp)
    else:
        pool_conc = [0] * len(ydata)

    return pool_volume, pool_conc


def purity_and_yield(inlet_profiles, outlet_profiles):
    # calculate cumulative mass*vol for all components
    cum_masses = []
    for comp in outlet_profiles:
        cum_mass = [sum(comp[:i]) for i in range(len(comp))]
        cum_masses.append(cum_mass)

    # calculate cumulative mass*vol loaded (from inlet profile) for each component
    in_masses = []
    for comp in inlet_profiles:
        in_mass = [sum(comp[:i]) for i in range(len(comp))]
        in_masses.append(in_mass)
        
    total_mass = [sum(j) for j in zip(*cum_masses)]

    cum_purity = []
    cum_yield = []
    for k in range(len(cum_masses[0])):
        if total_mass[k] > 0 and cum_masses[0][k] > 0:
            p = cum_masses[0][k] / total_mass[k]
            y = cum_masses[0][k] / in_masses[0][k]
        else:
            p = 0
            y = 0
        cum_purity.append(p)
        cum_yield.append(y)
    
    return cum_purity, cum_yield
