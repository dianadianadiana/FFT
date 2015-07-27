import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from pylab import *
from scipy.optimize import curve_fit
import os
import csv
from scipy.interpolate import LSQUnivariateSpline
import time

start_time = time.clock() #start the clock

################################################################################
################################################################################
################################ FUNCTIONS #####################################
################################################################################
################################################################################

############### GOAL: find all the values in the power array ###################
############# that are considered to be peaks and all the values ###############
########## in the power array that are considered to be harmonics ##############
    
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
    
def poly_maker(x, y, func, poly = 1):
    #z1 = z[0]*t**5 + z[1]*t**4 + z[2]*t**3 + z[3]*t**2 + z[4]*t + z[5]
    z = np.polyfit(x, y, poly)
    i = 0
    znew = np.zeros(len(func)) + z[poly]
    while i < poly:
        znew = znew + z[i]*func**(poly-i)
        i += 1
    return znew
    
################################################################################
############################## cluster algorithm ###############################
# goal: find out where the clusters are, and keep only the max of each cluster #
################################################################################

def cluster(index_arr, arr, peak_width = .015):
    """
        Takes an array of values (indexes) and clusters them (based on the width)
        Parameters:
            index_arr -- array of indexes (sub array from a bigger array)
            arr -- array of values (it is a sub array from a bigger array)
            peak_width -- how wide is the peak
        Returns:
            clusters -- a list of lists. each element of clusters is one cluster
    """
    clusters = []
    i = 0
    while i < len(index_arr):
        temp_arr = [index_arr[i]]
        j = i+1
        while j < len(index_arr) and (np.abs(arr[temp_arr[0]] - arr[index_arr[j]]) <= peak_width):
            temp_arr.append(index_arr[j])
            j+=1
        clusters.append(temp_arr)
        i=j
    return clusters
    
def cluster_max(clusters, y):
    """ 
        Takes a list of lists (clusters) and finds the index of where each max value of a cluster is
        Parameters:
            clusters -- a list of lists
            y -- the bigger array
        Returns:
            max_indexes -- an array of the indexes of where the max of each cluster is
    """
    max_indexes = np.empty(0)
    for elem in clusters:
        max_indexes = np.append(max_indexes, elem[y[elem].argmax()])
    return max_indexes.astype(int)

def peak_verifier(index_arr, arr1, arr2, n, peak_width):
    """
        The purpose of this function is to recognize if two points are on the same peak, 
            and if there are two poitns on the same peak, then the point that is the true
            max of the peak will be chosen, whereas the other one will be discarded.
        Parameters:
            index_arr: this is the peak_indexes or harmonics_indexes
            arr1: freq*constant
            arr2: power
            n: 2*n = the minumum number of data points for a peak
            peak_width: the width of the peak
        Returns:
            the index_arr with the indexes of the true peak maxes
    """
    k = 0
    delete_arr = []
    while k < len(index_arr) - 1:
        curr_index = index_arr[k]
        next_index = index_arr[k+1]
        if np.abs(arr1[curr_index]-arr1[next_index]) <= peak_width:
            curr_lower, curr_upper = curr_index - n, curr_index + n
            next_lower, next_upper = next_index - n, next_index + n
            if curr_lower < 0:
                curr_lower = 0
            if next_lower < 0:
                next_lower = 0
            #curr_maxpow and next_maxpow should be the same
            curr_maxpow = np.amax(arr2[curr_lower:curr_upper])
            #next_maxpow = np.amax(arr2[next_lower:next_upper])
            if arr2[curr_index] == curr_maxpow:
                delete_arr.append(k+1)
            else:
                delete_arr.append(k)  
        k+=1
    return np.delete(index_arr, delete_arr)

def limit_applier(arr, lower_limit = 1.0, upper_limit = 10.0):
    """
    Takes an array and a lower and upper limit and returns an array of the indices that 
        are below the lower limit or higher than the upper limit -- it is the delete_arr
    """
    delete_arr = np.where(arr < lower_limit)[0]
    delete_arr = np.append(delete_arr, np.where(arr > upper_limit)[0])
    return delete_arr

################################################################################
################################################################################
############################# read all the data ################################
################################################################################
################################################################################

information_arr =[] # each elem has this format [ filename, code #, peak freq, period ]

path2 = "C:/Users/dianadianadiana/Desktop/Research/Field2/Decorrelatedphotometry2/"
path3 = "C:/Users/dianadianadiana/Desktop/Research/Field3/Decorrelatedphotometry2/Decorrelatedphotometry2" # 16,339 Files
datapath = "C:/Users/dianadianadiana/Desktop/Research/Data/LC/" # 16 files
temppath = "C:/Users/dianadianadiana/Desktop/temp/LC/"
#beginning = "LCfluxesepic"
#ending = "star00"

# array of all the epic numbers
# 205029914: period of 4.98 - short frequencies
# 204129699: check!!! should have freq around 1.5
# 205900901: 17 hrs in the temp file!
# new data set: 205899208 look at this one!!!
#filename_arr = ["205900901"]

path = temppath
filename_arr = os.listdir(path)

for epicname in filename_arr[:]:
    #chosenfile = "C:/Users/dianadianadiana/Desktop/Research/Data/Data/LCfluxesepic" + str(filename) + "star00"
    chosenfile = path + str(epicname)
    data = np.loadtxt(chosenfile)
    data = np.transpose(data)
    t = data[0] # in days
    f = data[1] # flux

################################################################################
################################################################################
## Determine the appropriate cadence to which the data should be interpolated ##
######################### Get the GAP array ####################################
################################################################################
################################################################################

    num_time = len(t) # the original number of data time points (w gaps)
    
    delta_t = t[1:num_time]-t[0:num_time-1] # has a length = len(t) - 2 (bc it ignores endpoints)
    cadence = np.median(delta_t) # what the time interval between points with gaps currently is
    #Fill gaps in the time series (according to 2012 thesis method)
    gap = delta_t/cadence # has a length = len(t) - 2 (bc it ignores endpoints)
    gap = np.around(gap)
    gap = np.append(gap, 1)
    
    gap_cut = 1.1   #Time differences greater than gap_cut*cadence are gaps    
    gap_loc = [i for i in np.arange(len(gap)) if (gap[i] > gap_cut)]
    num_gap = len(gap_loc) # the number of gaps present 
    biggest_gap = np.amax(gap)

################################################################################
########## Create a new time and flux array to account for the gaps ############
################################################################################

    num_cad = sum(gap)  # the number of data points
    time_cad = np.arange(num_cad)*cadence + np.amin(t)
    flux_cad = np.arange(num_cad, dtype = np.float) 
        
    oldflux_st = 0 # this is an index
    newflux_st = 0 # this is an index
    
    for n in np.arange(num_gap):
        oldflux_ed = gap_loc[n]
        gap_sz     = gap[gap_loc[n]]
        newflux_ed = newflux_st + (oldflux_ed-oldflux_st)

        flux_cad[newflux_st:newflux_ed] = f[oldflux_st:oldflux_ed]
        
        gap_fill = (np.ones(gap_sz-1))*f[oldflux_ed]
        
        flux_cad[newflux_ed+1:newflux_ed+gap_sz] = gap_fill
        flux_cad[newflux_ed] = np.mean([flux_cad[newflux_ed-1], flux_cad[newflux_ed+1]])
        
        oldflux_st = oldflux_ed + 1
        newflux_st = newflux_ed + gap_sz
    
    #account for last part where there is no gap after
    flux_cad[newflux_st:num_cad] = f[oldflux_st:num_time] 
        
################################################################################
################################################################################
#################################### FFT part ##################################
################################################################################
################################################################################

    # oversampling
    time_cad *= 24. #in hours
    N = len(time_cad)
    N_log = np.log2(N) # 2 ** N_log = N
    exp = np.round(N_log)
    if exp < N_log:
        exp += 1 #compensate for if N_log was rounded down
    
    extra_fact = 5
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
    peak_width_to_zero = bin_sz * 2**extra_fact
    peak_width = 2 * peak_width_to_zero

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
    
    #fit1 = np.median(power_fit)
    #fit2 = poly_maker(freq_fit, power_fit, freq, poly = 2)
    first_knot_fit_i = np.where((freq_fit*constant - freq_fit[0]*constant) >= knot_w)[0][0] #index of the first knot is the first point that is knot_w away from the first value of x 
    last_knot_fit_i = np.where((freq_fit[-1]*constant-freq_fit*constant) >= knot_w)[0][-1]#index of the last knot is the first point that is knot_w away from the last value of x
    knots_fit = np.arange(freq_fit[first_knot_fit_i]*constant, freq_fit[last_knot_fit_i]*constant,knot_w)
    spline = LSQUnivariateSpline(freq_fit*constant,power_fit,knots_fit, k=2) #the spline, it returns the piecewise function
    fit3 = spline(freq*constant) #the actual y values of the fit
    if np.amin(fit3) < 0:
        fit3 = np.ones(len(freq))
    
    # relative power
    power_rel = pre_power_rel / fit3

################################################################################
################################################################################
############################### Peak Limits ####################################
################################################################################   
################################################################################

    peak_constraint, harmonics_constraint = 4.6, 3.0
    
    val_indexes = constraint_index_finder(peak_constraint, freq, power_rel) #peaks
    val_indexes1 = constraint_index_finder(harmonics_constraint, freq, power_rel) #harmonics

    peak_indexes = cluster_max(cluster(val_indexes, freq*constant, peak_width), power_rel)    
    harmonics_indexes = cluster_max(cluster(val_indexes1, freq*constant, peak_width), power_rel)  
    
    peak_indexes = peak_verifier(peak_indexes, freq*constant, power_rel, n, peak_width)
    harmonics_indexes = peak_verifier(harmonics_indexes, freq*constant, power_rel, n, peak_width) 

    # keep all of the original peak_indexes/harmonics_indexes to check later on 
    # if it's a longer period planet
    highest_period = 150.0 # in hours
    lowest_freq = 24. / highest_period # in cycles per day
    original_peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes]*constant,lowest_freq))
    original_harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes]*constant,lowest_freq))
    
    # we only want peaks that are between freqs of [1,10] cycles/day
    lower_freq, upper_freq = 1.0, 10.0
    peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes]*constant))
    harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes]*constant))
    
################################################################################
################################################################################
############ Determining potential periods based on the FFT ####################
################################################################################
################################################################################
    
    potential_arr = []
    
    for elem in peak_indexes:
        number = len(harmonics_indexes) + 1
        poss_indexes = np.arange(2, number) * elem - 1 #possible indexes
        poss_indexes_lower = poss_indexes - n # lower bound of possible indexes
        poss_indexes_upper = poss_indexes + n # upper bound of possible indexes
        poss_indexes_bound = np.array([poss_indexes_lower, poss_indexes_upper])
        poss_indexes_bound = np.transpose(poss_indexes_bound)
        temp_arr = [elem]
        for elem1 in harmonics_indexes:
            for lower, upper in poss_indexes_bound:
                if elem1 >= lower and elem1 <= upper:
                    temp_arr.append(elem1)
                    break # break because no elem1 will only satisfy this condition
                elif lower > elem1:
                    break # won't ever satisfy the condition
        potential_arr.append(temp_arr)
        
    rel_power_sums = []
    for elem in potential_arr:
        if len(elem) == 1: # if only a peak was detected with no harmonics
            rel_power_sums.append(0) 
        else: # else sum up all the relative power
            rel_power_sums.append(np.sum(power_rel[elem])) 
            
    # booleans
    has_peaks = len(peak_indexes) > 0
    longer_period = False
    # good peak means that there is at least one peak with one other harmonic
    good_peak = has_peaks and np.amax(rel_power_sums) > 0
       
################################################################################
################# checking if it might be longer period planet #################
################################################################################

    if has_peaks and good_peak:
        relevant_index = potential_arr[np.argmax(rel_power_sums)][0]
        
        ### this 'for loop' goes through the peaks whose freq are [lowest_freq,1]
        ### and checks to see if one of those freqs could potentially be
        ### the relevant freq
        potential_indexes_longer_period = [] # indicies of potential revelant freqs
        for elem in original_peak_indexes: 
            number = len(harmonics_indexes) + 1
            poss_indexes = 1. / np.arange(2, number) * relevant_index - 1
            poss_indexes_lower = poss_indexes - n # lower bound of possible indexes
            poss_indexes_upper = poss_indexes + n # upper bound of possible indexes
            poss_indexes_bound = np.array([poss_indexes_lower, poss_indexes_upper])
            poss_indexes_bound = np.transpose(poss_indexes_bound)
            for lower, upper in poss_indexes_bound:
                if elem >= lower and elem <= upper:
                    potential_indexes_longer_period.append(elem)
                    break # break because elem only satisfies this condition
                elif upper > elem1:
                    break # won't ever satisfy the condition
                    
        ### this for loop goes through the freqs in the interval [lowest_freq,1] 
        ### that may potentially be the relevant freq
        ### follows exact algorithm as the for loop that goes through each peak
        ### in peak_indexes and tries to see if they are potentially good freq
        potential_arr1 =[]
        for elem in potential_indexes_longer_period:
            number = len(original_harmonics_indexes) + 1
            poss_indexes = np.arange(2, number) * elem - 1
            poss_indexes_lower = poss_indexes - n # lower bound of possible indexes
            poss_indexes_upper = poss_indexes + n # upper bound of possible indexes
            poss_indexes_bound = np.array([poss_indexes_lower, poss_indexes_upper])
            poss_indexes_bound = np.transpose(poss_indexes_bound)
            temp_arr = [elem]
            for elem1 in harmonics_indexes: # or original_harmonics_indexes?
                for lower, upper in poss_indexes_bound:
                    if elem1 >= lower and elem1 <= upper:
                        temp_arr.append(elem1)
                        break # break because elem1 only satisfies this condition
                    elif lower > elem1:
                        break # won't ever satisfy the condition
            potential_arr1.append(temp_arr)
            
        if len(potential_arr1)>0:
            rel_power_sums1 = []
            for elem in potential_arr1:
                if len(elem) == 1:
                    rel_power_sums1.append(0)
                else:
                    rel_power_sums1.append(np.sum(power_rel[elem]))
            
            if np.amax(rel_power_sums1) > np.amax(rel_power_sums):
                longer_period = True
                relevant_index = potential_arr1[np.argmax(rel_power_sums1)][0]
        
        relevant_freq = freq[relevant_index]*constant
        relevant_period = 24./relevant_freq

################################################################################
################################################################################
############################ SAVING THE FIGURE #################################
################################################################################
################################################################################

    fig = plt.figure(figsize=(20,15))
    
    ax1 = fig.add_subplot(211)
    time_cad /= 24
    ax1.scatter(time_cad, flux_cad, s=10, c='black')
    ax1.plot(time_cad, flux_cad, 'black', linewidth = .75)
    plt.title("Lightcurve " + str(epicname), fontsize = 16)
    plt.xlabel("Time (Days)")
    plt.ylabel("Numerical Flux")
    plt.xlim([np.amin(time_cad),np.amax(time_cad)])
    delta = np.amax(flux_cad) - np.amin(flux_cad)
    plt.ylim([1 - 1.5 * delta, 1. + .5 * delta])
    
    ax2 = fig.add_subplot(223)
    ax2.plot(freq*constant, power, 'black')
    if has_peaks:
        ax2.scatter(freq[peak_indexes]*constant, power[peak_indexes], s=30, c="black")
        plt.title("Numfreq = " + str(len(peak_indexes)), fontsize = 16)
    else:
        plt.title("NO PEAKS DETECTED")
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("Amplitude")
    plt.xlim([0,10])
    plt.ylim(bottom=0)
    
    ax3 = fig.add_subplot(224)
    ax3.plot(freq*constant, power_rel,'black')
    if has_peaks and good_peak:
        ax3.scatter(freq[relevant_index]*constant, power_rel[relevant_index], c='black', s=50)
        plt.title("PEAK FREQ = " + str(relevant_freq) + " Period: " + str(relevant_period), fontsize =16)
        #upper  = np.round(relevant_freq)
        #if upper < relevant_freq:
        #    upper += 1
    else:
        plt.title("NO PEAKS DETECTED")
        #upper = 5.0
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("Relative Amplitude")
    upper = 10
    plt.xlim([0, upper])
    plt.ylim([0, 1.5*np.amax(power_rel[n:])])
    
    plt.show()
    
    if not has_peaks:
        info = [epicname, 1, 0, 0]#, 'WARNING: no peaks detected']
    elif has_peaks and not good_peak:
        info = [epicname, 2, 0, 0]#, 'WARNING: a peak with no harmonics']
    elif has_peaks and good_peak and harmonics_indexes[0] < peak_indexes[0]:
        info = [epicname, 3, relevant_freq, relevant_period]#,'WARNING: there may be a harmonic peak that came before the first main peak']
    elif longer_period:
        info = [epicname, 4, relevant_freq, relevant_period]#,'WARNING: may be longer period']
    elif biggest_gap/len(time_cad) >= .2:
        info = [epicname, 5, relevant_freq, relevant_period]#,'WARNING: huge gap']
    else:
        info = [epicname, 0, relevant_freq, relevant_period]

    information_arr.append(info)
    
    #graphpath3 = "C:/Users/dianadianadiana/Desktop/Research/Field3/Figures"
    epicname_no_txt = epicname[:len(epicname) - 4]
    figurepath = "C:/Users/dianadianadiana/Desktop/Research/Data/Figures/Figures_Spline/"
    #figurepath = "C:/Users/dianadianadiana/Desktop/temp/Figures/"
    #figurepath = "C:/Users/dianadianadiana/Desktop/Research/Field2/Figures/"
    #fig.savefig(figurepath + str(epicname_no_txt) + "_figure.png", dpi = 300)
    
#for elem in information_arr:
#    print(elem)
    
############################ Saving the Information ############################
#http://gis.stackexchange.com/questions/72458/export-list-of-values-into-csv-or-txt-file

#res = information_arr
#csvfile = figurepath + "information.txt"
#
##Assuming res is a flat list
#with open(csvfile, "w") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    for val in res:
#        writer.writerow([val])    
#
##Assuming res is a list of lists
#with open(csvfile, "w") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    writer.writerows(res)
    
print time.clock() - start_time, "seconds"
