import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import LSQUnivariateSpline

def constraint_index_finder(constraint, x, y, poly = 0):
    """ 
    Takes two arrays (freq and power) and then computes the line of best fit/mean value (using poly)
    The value is then multiplied with the constraint, and any value in y (power) that is above
        constraint * mean value is stored to an array --> That array is then returned
    Return value: 
        val_indexes -- array with indexes where y[indexes] output values greater than 
                    constraint * mean value
    
    constraint: the number times the best fit/mean of y
    x: x values -- freq
    y: y values -- power
    poly: the power of the best line fit
    
    """
    # raise error if x and y don't have the same length
    if len(x) != len(y):
        raise Exception('x and y needs to have the same length.')

    z = np.polyfit(x, y, poly)
    val = z[0]
    return np.where(y >= constraint* val)[0]

################################################################################
################################################################################
#################################### FFT part ##################################
################################################################################
################################################################################

def fft_part(time_cad, flux_cad, extra_fact):
    # oversampling
    time_cad *= 24. #in hours
    N = len(time_cad)
    N_log = np.log2(N) # 2 ** N_log = N
    exp = np.round(N_log)
    if exp < N_log:
        exp += 1 #compensate for if N_log was rounded down
    
    newN = 2**(exp + extra_fact)
    n = np.round(newN/N)
    diff = newN - N
    mean = np.median(flux_cad)
    voidf = np.zeros(diff) + mean
    newf = np.append(flux_cad, voidf)
    
    norm_fact = 2.0 / newN # normalization factor 
    f_flux = fft(newf) * norm_fact
        
    freq = fftfreq((len(newf)))
    d_pts = (np.amax(time_cad) - np.amin(time_cad)) / N
    freq_fact = 1.0 / d_pts #frequency factor 
    
    postivefreq = freq > 0 # take only positive values
    freq, f_flux = freq[postivefreq], f_flux[postivefreq]
    
    power = np.abs(f_flux)
    
    conv_hr_day = 24. #conversion factor from cycles/hour to cycles/day
    constant = freq_fact*conv_hr_day
    
    bin_sz = 1./len(newf) * constant
    #peak_width_to_zero = bin_sz * 2**extra_fact
    #peak_width = 2 * peak_width_to_zero

################################################################################
########################### Normalizing the FFT ################################
################################################################################
    
    knot_w = 30*n*bin_sz # difference in freqs between each knot
    
    # first fit
    first_knot_i = np.where((freq*constant - freq[0]*constant) >= knot_w)[0][0] #index of the first knot is the first point that is knot_w away from the first value of x 
    last_knot_i = np.where((freq[-1]*constant-freq*constant) >= knot_w)[0][-1]#index of the last knot is the first point that is knot_w away from the last value of x
    knots = np.arange(freq[first_knot_i]*constant, freq[last_knot_i]*constant,knot_w)
    spline = LSQUnivariateSpline(freq*constant, power, knots, k=2) #the spline, it returns the piecewise function
    fit = spline(freq*constant) #the actual y values of the fit
    if np.amin(fit) < 0:
        fit = np.ones(len(freq))
    pre_power_rel = power/fit
    
    # second fit -- by deleting the points higher than 4 times the average value of pre_power_rel
    pre_indexes = constraint_index_finder(4, freq, pre_power_rel) #peaks
    power_fit = np.delete(pre_power_rel, pre_indexes)
    freq_fit = np.delete(freq, pre_indexes)
    
    first_knot_fit_i = np.where((freq_fit*constant - freq_fit[0]*constant) >= knot_w)[0][0] #index of the first knot is the first point that is knot_w away from the first value of x 
    last_knot_fit_i = np.where((freq_fit[-1]*constant-freq_fit*constant) >= knot_w)[0][-1]#index of the last knot is the first point that is knot_w away from the last value of x
    knots_fit = np.arange(freq_fit[first_knot_fit_i]*constant, freq_fit[last_knot_fit_i]*constant,knot_w)
    spline = LSQUnivariateSpline(freq_fit*constant,power_fit,knots_fit, k=2) #the spline, it returns the piecewise function
    fit1 = spline(freq*constant) #the actual y values of the fit, apply it back to freq
    if np.amin(fit1) < 0:
        fit1 = np.ones(len(freq))
    
    # relative power
    power_rel = pre_power_rel / fit1
    return freq, power_rel, power
