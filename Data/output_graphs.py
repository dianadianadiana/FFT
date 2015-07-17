import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from pylab import *
from scipy.optimize import curve_fit

################################################################################
################################ FUNCTIONS #####################################
################################################################################

### GOAL: find all the values in the power array that are considered to be peaks
###    and all the values in the power array that are considered to be harmonics
    
def constraint_index_finder(constraint, x, y, poly = 0, loops = 10):
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
    loops: number of times the mean will be getting recalculated
    
    """
    # raise error if x and y don't have the same length
    if len(x) != len(y):
        raise Exception('x and y needs to have the same length.')
                        
    # val_indexes: stores all the indexes in y that have 
    # a power value greater constraint * value of best line fit
    val_indexes= np.empty(0) 
    
    j = 0
    while j < loops:
        z = np.polyfit(x, y, poly)
        if j > 0 and z[0] == val:
            break # to leave the loop early if the same value is brought up
        val = z[0]
        i = 0
        while i < len(y):
            if y[i] >= constraint * val:
                val_indexes = np.append(val_indexes, i)
            i += 1
        j += 1
    val_indexes = val_indexes.astype(int)
    return val_indexes
    
def poly_maker(x, y, func, poly = 1):
    #z1 = z[0]*t**5 + z[1]*t**4 + z[2]*t**3 + z[3]*t**2 + z[4]*t + z[5]

    z = np.polyfit(x, y, poly)
    i = 0
    znew = np.zeros(len(func)) + z[poly]
    while i < poly:
        znew = znew + z[i]*func**(poly-i)
        i += 1
    return znew
    
############################## cluster algorithm ##############################
# goal: find out where the clusters are, and keep only the max of each cluster

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
            y -- the bigger array (with length newN)
        Returns:
            max_indexes -- an array of the indexes of where the max of each cluster is
    """
    max_indexes = np.empty(0)
    for elem in clusters:
        max_indexes = np.append(max_indexes, elem[y[elem].argmax()])
    max_indexes = max_indexes.astype(int)
    return max_indexes

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
            next_maxpow = np.amax(arr2[next_lower:next_upper])

            if power[curr_index] == curr_maxpow:
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
    delete_arr = np.empty(0)
    i = 0
    while i < len(arr):
        if arr[i] < lower_limit:
            delete_arr = np.append(delete_arr, i)
        elif arr[i] > upper_limit:
            delete_arr = np.append(delete_arr, np.arange(i, len(arr)))
            break 
        i += 1
    delete_arr = delete_arr.astype(int)
    return delete_arr

################################################################################
############################# read all the data ################################
################################################################################

#filename_arr = ['201124933', '201127583', '201140274', '201143076', '201149315', '201159747',
#                '201167435', '201174877', '201345483', '201637175', '201862715', '202060551',
#                '202068113', '203099398', '205029914', '205071984']

information_arr =[""] # each elem has this format [ filename, peak freq, period, notes (optional)]

path = "C:/Users/dianadianadiana/Desktop/Decorrelatedphotometry2/Decorrelatedphotometry2/"
beginning = "LCfluxesepic"
ending = "star00"

# array of all the epic numbers
filename_arr = []

for epicname in filename_arr:
    #chosenfile = "C:/Users/dianadianadiana/Desktop/Research/Data/Data/LCfluxesepic" + str(filename) + "star00"
    chosenfile = path + beginning + str(epicname) + ending
    data = np.loadtxt(chosenfile + ".txt")
    data = np.transpose(data)
    t = data[0] # in days
    f = data[1] # flux

################################################################################
## Determine the appropriate cadence to which the data should be interpolated ##
######################### Get the GAP array ####################################
################################################################################

    num_time = len(t) # the original number of data time points (w gaps)
    
    delta_t = t[1:num_time-1]-t[0:num_time-2] # has a length = len(t) - 2 (bc it ignores endpoints)
    cadence = np.mean(delta_t) # what the time interval between points with gaps currently is
    #Fill gaps in the time series (according to 2012 thesis method)
    gap = delta_t/cadence # has a length = len(t) - 2 (bc it ignores endpoints)
    gap = np.around(gap, decimals = 0)
    
    gap_cut = 1.1   #Time differences greater than gap_cut*cadence are gaps
    gap_loc = np.empty(0) # array that holds the indexes of where the gaps are
    
    if gap[0] == 0:
        gap += 1
        
    i = 0
    while i < len(gap):
        if gap[i] > gap_cut:
            gap_loc = np.append(gap_loc, i)
        i += 1
    num_gap = len(gap_loc) # the number of gaps present 
    biggest_gap = np.amax(gap)

################################################################################
########## Create a new time and flux array to account for the gaps ############
################################################################################
    num_cad = sum(gap) + 2#+1 # the number of data points
    time_cad = np.arange(num_cad)*cadence + np.amin(t)
    flux_cad = np.arange(num_cad, dtype = np.float) 
        
    oldflux_st = 0 # this is an index
    newflux_st = 0 # this is an index
    n = 0
    while n < num_gap :
        oldflux_ed = gap_loc[n]
        gap_sz     = gap[gap_loc[n]]
        newflux_ed = newflux_st + (oldflux_ed-oldflux_st)
        if len(flux_cad[newflux_st:newflux_ed]) != len(f[oldflux_st:oldflux_ed]):
            print(chosenfile)
            n = num_gap
        flux_cad[newflux_st:newflux_ed] = f[oldflux_st:oldflux_ed]
        
        gap_fill = (np.ones(gap_sz-1))*f[oldflux_ed]
        
        flux_cad[newflux_ed+1:newflux_ed+gap_sz] = gap_fill
        flux_cad[newflux_ed] = np.mean([flux_cad[newflux_ed-1], flux_cad[newflux_ed+1]])
        
        oldflux_st = oldflux_ed + 1
        newflux_st = newflux_ed + gap_sz
        n+=1 
    
    #account for last part where there is no gap after
    flux_cad[newflux_st:num_cad] = f[oldflux_st:num_time] 
        
################################################################################
#################################### FFT part ##################################
################################################################################

    # oversampling
    time_cad *= 24. #in hours
    N = len(time_cad)
    N_log = np.log2(N) # 2 ** N_log = N
    exp = np.round(N_log)
    if exp < N_log:
        exp += 1 #compensate for if N_log was rounded down
    
    extra_fact = 3
    newN = 2**(exp + extra_fact)
    n = newN/N
    n = np.round(n)
    diff = newN - N
    mean = np.mean(f)
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
    
    peak_error = .11 # should i do this?
    peak_width_error = peak_width + peak_width * peak_error

################################################################################
########################### Normalizing the FFT ################################
################################################################################
   
    def func(x, a, c, d):
            return a*np.exp(-c*(x))+d

    popt, pcov = curve_fit(func, freq*constant, power)
    z=func(freq*constant,*popt)
    pre_power_rel = power/z
   
    pre_indexes = constraint_index_finder(4, freq, pre_power_rel) #peaks
    power_fit = np.delete(pre_power_rel, pre_indexes)
    freq_fit = np.delete(freq, pre_indexes)    
    popt1, pcov1 = curve_fit(func, freq_fit*constant, power_fit)
    z1=func(freq_fit*constant,*popt)
   
    # Relative power    
    popt2, pcov2 = curve_fit(func, freq_fit*constant, power_fit)
    z2=func(freq*constant,*popt2)
    
    power_rel = pre_power_rel/z2
   
################################################################################
################################################################################
################################################################################   
   
    peak_constraint, harmonics_constraint = 4.0,3.0
    
    val_indexes = constraint_index_finder(peak_constraint, freq, power_rel) #peaks
    val_indexes1 = constraint_index_finder(harmonics_constraint, freq, power_rel) #harmonics

    peak_indexes = cluster_max(cluster(val_indexes, freq*constant, peak_width), power_rel)    
    harmonics_indexes = cluster_max(cluster(val_indexes1, freq*constant, peak_width), power_rel)  
    
    peak_indexes = peak_verifier(peak_indexes, freq*constant, power_rel, n, peak_width)
    harmonics_indexes = peak_verifier(harmonics_indexes, freq*constant, power_rel, n, peak_width)  
      
############################### peak limits ####################################
    # keep all of the original peak_indexes to check later on if it's a longer
    # period planet
    original_peak_indexes = peak_indexes
    original_harmonics_indexes = harmonics_indexes
    
    # we only want peaks that are between freqs of [1,10]    
    lower_freq, upper_freq = 1.0, 10.0
    peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes]*constant))
    harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes]*constant))
    
################################################################################
############ Determining potential periods based on the FFT ####################
################################################################################
    
    if len(peak_indexes) == 0 and len(harmonics_indexes) == 0:
        info = [epicname, None, None, 'WARNING: no peaks detected']
        information_arr.append(info)
        continue
    
    potential_arr = []
    
    if len(peak_indexes)>0:
        chosen_indexes = peak_indexes
    else:
        chosen_indexes = harmonics_indexes
    for elem in chosen_indexes:
        number = len(harmonics_indexes) + 1
        poss_indexes = np.arange(2, number) * elem - 1
        poss_indexes_lower = poss_indexes - n
        poss_indexes_upper = poss_indexes + n
        poss_indexes_bound = np.array([poss_indexes_lower, poss_indexes_upper])
        poss_indexes_bound = np.transpose(poss_indexes_bound)
        temp_arr = [elem]
        for elem1 in harmonics_indexes:
            for lower, upper in poss_indexes_bound:
                if elem1 >= lower and elem1 <= upper:
                    temp_arr.append(elem1)
                    break # break because no other value will satisfy the condition
                elif lower > elem1:
                    break # won't ever satisfy the condition
        potential_arr.append(temp_arr)
        
    rel_power_sums = []
    for elem in potential_arr:
        rel_power_sums.append(np.sum(power_rel[elem]))
        
###############################################################################
###############################################################################

    relevant_index = potential_arr[np.argmax(rel_power_sums)][0]
    
################# checking if it might be longer period planet #################

    potential_arr_longer_period = []
    for elem in original_peak_indexes: #maybe should be original harmonics indexes?
        number = len(harmonics_indexes) + 1
        poss_indexes = 1. / np.arange(2, number) * relevant_index - 1
        poss_indexes_lower = poss_indexes - n
        poss_indexes_upper = poss_indexes + n
        poss_indexes_bound = np.array([poss_indexes_lower, poss_indexes_upper])
        poss_indexes_bound = np.transpose(poss_indexes_bound)
        for lower, upper in poss_indexes_bound:
            if elem >= lower and elem <= upper:
                potential_arr_longer_period.append(elem)
                break # break because no other value will satisfy the condition
            elif upper > elem1:
                break # won't ever satisfy the condition
        
    potential_arr1 =[]
    for elem in potential_arr_longer_period:
        number = len(original_harmonics_indexes) + 1
        poss_indexes = np.arange(2, number) * elem - 1
        poss_indexes_lower = poss_indexes - n
        poss_indexes_upper = poss_indexes + n
        poss_indexes_bound = np.array([poss_indexes_lower, poss_indexes_upper])
        poss_indexes_bound = np.transpose(poss_indexes_bound)
        temp_arr = [elem]
        for elem1 in harmonics_indexes:
            for lower, upper in poss_indexes_bound:
                if elem1 >= lower and elem1 <= upper:
                    temp_arr.append(elem1)
                    break # break because no other value will satisfy the condition
                elif lower > elem1:
                    break # won't ever satisfy the condition
        potential_arr1.append(temp_arr)
        
    longer_period = False
    if len(potential_arr1)>0:
        rel_power_sums1 = []
        for elem in potential_arr1:
            rel_power_sums1.append(np.sum(power_rel[elem]))
        if np.amax(rel_power_sums1) > np.amax(rel_power_sums):
            longer_period = True
            relevant_index = potential_arr1[np.argmax(rel_power_sums1)][0]
    
    relevant_freq = freq[relevant_index]*constant
    relevant_period = 24./relevant_freq

################################################################################
############################ SAVING THE FIGURE #################################
################################################################################

    fig = plt.figure(figsize=(20,15))
    
    ax1 = fig.add_subplot(211)
    time_cad /= 24
    ax1.scatter(time_cad,flux_cad, s= 10)
    ax1.plot(time_cad,flux_cad, 'black', linewidth = .75)
    plt.title("Lightcurve " + str(chosenfile), fontsize = 16) #or filename
    plt.xlabel("Time (Days)")
    plt.ylabel("Numerical Flux")
    plt.xlim([np.amin(time_cad),np.amax(time_cad)])
    delta = np.amax(flux_cad) - np.amin(flux_cad)
    plt.ylim([1 - 1.5*delta, 1. + .5*delta])
    
    ax2 = fig.add_subplot(223)
    ax2.plot(freq*constant, power_rel,'black')
    ax2.scatter(freq[peak_indexes]*constant,power_rel[peak_indexes], s =30, c ="black")
    plt.title("Numfreq = " + str(len(peak_indexes)), fontsize =16)
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("Relative Amplitude")
    plt.xlim([0,10])
    plt.ylim(bottom=0)
    
    ax3 = fig.add_subplot(224)
    ax3.plot(freq*constant, power_rel,'black')
    ax3.scatter(freq[relevant_index]*constant, power_rel[relevant_index], c='black', s=50)
    plt.title("PEAK FREQ = " + str(relevant_freq) + " Period: " + str(relevant_period), fontsize =16)
    plt.xlabel("Frequency (cycles/day)")
    upper  = np.round(relevant_freq)
    if upper < relevant_freq:
        upper += 1
    plt.xlim([0, upper])
    plt.ylim(bottom=0)
    
    #plt.show()
    
    if biggest_gap/len(time_cad) >= .2:
        info = [epicname, relevant_freq, relevant_period,'WARNING: huge gap']
        information_arr.append(info)
        continue
    elif longer_period:
        info = [epicname, relevant_freq, relevant_period,'WARNING: may be longer period']
        information_arr.append(info)
        continue
    elif len(peak_indexes) != 0 and harmonics_indexes[0] < peak_indexes[0]:
        info = [epicname, relevant_freq, relevant_period,'WARNING: there may be a harmonic peak that came before the first main peak']
        information_arr.append(info)
        continue
        
    info = ["LCfluxesepic" + str(filename) + "star00", relevant_freq, relevant_period]
    information_arr.append(info)
    
    #fig.savefig(chosenfile + "_testgraphs.png", dpi = 300)
    
    
for elem in information_arr:
    print(elem)