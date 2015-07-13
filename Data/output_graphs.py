import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

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
    
##### cluster algorithm #####
# goal: find out where the clusters are, and keep only the max of each cluster

def cluster(arr, num_indexes, width = .005):
    """
        Takes an array of values (indexes) and clusters them (based on the width)
        Parameters:
            arr -- array of values (it is a sub array from a bigger array)
            num_indexes -- the length of the bigger array (newN)
            width -- how big do we want the cluster to account for
        Returns:
            clusters -- a list of lists. each element of clusters is one cluster
    """
    clusters = []
    i = 0
    while i < len(arr):
        temp_arr = []
        j = i+1
        temp_arr.append(arr[i])
        while j < len(arr) and (np.abs(arr[j-1] - arr[j]) <= width * num_indexes):
            temp_arr.append(arr[j])
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

#%cd "C:/Users/dianadianadiana/Desktop/Research/Data/Data"

filename_arr = ['201124933', '201127583', '201140274', '201143076', '201149315', '201159747',
                '201167435', '201174877', '201345483', '201637175', '201862715', '202060551',
                '202068113', '203099398', '205029914', '205071984']
information_arr =[] # [ filename, peak freq, period]
for filename in filename_arr:
    chosenfile = "C:/Users/dianadianadiana/Desktop/Research/Data/Data/LCfluxesepic" + str(filename) + "star00"
    data = np.loadtxt(chosenfile + ".txt")
    data = np.transpose(data)
    t = data[0] # in days
    t = t*24. # in hours
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
    N = len(t)
    N_log = np.log2(N) # 2 ** N_log = N
    N_over = np.round(N_log)
    if N_over < N_log:
        N_over += 1 #compensate for if N_log was rounded down
    
    newN = 2**(N_over)
    diff = newN - N
    mean = np.mean(f)
    voidf = np.zeros(diff) + mean
    newf = np.append(f, voidf)
    
    norm_fact = 2.0 / newN # normalization factor 
    f_flux = fft(f) * norm_fact
        
    freq = fftfreq(np.int(len(newf)))
    d_pts_new = (np.amax(t)-np.amin(t))/newN
    freq_fact = 1.0 / d_pts_new #frequency factor 
    
    postivefreq = freq > 0 # take only positive values
    freq, f_flux = freq[postivefreq], f_flux[postivefreq]
    
    power = np.abs(f_flux)
    
    conv_hr_day = 24. #conversion factor from cycles/hour to cycles/day
    constant = freq_fact*conv_hr_day

################################################################################
################################################################################
################################################################################
   
    peak_constraint, harmonics_constraint = 4.0, 3.0
    
    val_indexes = constraint_index_finder(peak_constraint, freq, power) #peaks
    val_indexes1 = constraint_index_finder(harmonics_constraint, freq, power) #harmonics

    power_fit = np.delete(power, val_indexes)
    freq_fit = np.delete(freq, val_indexes)
    z1 = poly_maker(freq_fit, power_fit, freq_fit, poly = 1)
    
    # Relative power
    z2 = poly_maker(freq_fit, power_fit, freq)
    power_rel = power/z2

    peak_indexes = cluster_max(cluster(val_indexes, newN), power)    
    harmonics_indexes = cluster_max(cluster(val_indexes1, newN), power)    

    ## peak limits
    # we only want peaks that are between freqs of [1,10]    
    original_peak_indexes = peak_indexes
    original_harmonics_indexes = harmonics_indexes
    
    lower_freq, upper_freq = 1.0, 10.0
    peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes]*constant))
    harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes]*constant))

    # now we want to delete the peak values that are not the absolute maxima within the cluster
    # and we just want to keep the absolute maxima
    
    delete_arr =  [i for i in val_indexes if (i not in peak_indexes)]
    freq_delete = np.delete(freq, delete_arr)
    power_delete = np.delete(power, delete_arr)
    
################################################################################
############ Determining potential periods based on the FFT ####################
################################################################################

    # the way potential arr is set up as a list
    # like [[0 1 2 3 4 5 6] [1 3 5 7] [2 5 8]]
    # in each element of the potential_arr, the starting index refers to index in peak_indexes
    # and the other indexes refer the the indexes in harmonics_indexes
    potential_arr = []
    #**remember** constant = freq_fact*conv_hr_day
    
    if len(peak_indexes) == 0:
        #print(chosenfile)
        #print('No peaks have been detected -- may be a longer period planet')
        info = ["LCfluxesepic" + str(filename) + "star00", None, None]
        information_arr.append(info)
        continue
    
    i = 0
    while i < len(peak_indexes): #iterate through all the "main" peaks (higher than 4.6 times the mean)
        curr_freq = freq[peak_indexes[i]]*constant # get the frequency of the current index
        number = len(harmonics_indexes)
        # set up an array whose values are multiples of the current frequency
        curr_freq_values = np.arange(1,number)*curr_freq 
        # error is how close do we want the multiple of the current frequency to be to the 
        error = .1 #better way of setting this value?
        
        temp_arr = [i] # set up an array starting with the current index
        #iterate through all the harmonics where peaks were 3 times the mean
        j = i + 1 # j is one greater than i so it looks at the next value, and not the same one
        while j < len(harmonics_indexes): 
            # the frequency that will be compared to the the multiples of the current frequency
            other_freq = freq[harmonics_indexes[j]]*constant 
        
            #iterate through curr_freq_values and compare the other_freq to each multiple of curr_freq
            for multiple in curr_freq_values:
                if np.abs(other_freq - multiple) <= error: 
                    # add the index from the harmonics array 
                    temp_arr.append(j) 
                    break # break because no other value will satisfy the condition
                elif multiple > other_freq:
                    break # no point in trying to satisfy the condition
            j+=1   
        potential_arr.append(temp_arr)
        i+=1
        
    rel_power_sums = np.empty(0)
    for elem in potential_arr:
        total =  power_rel[peak_indexes[elem[0]]]
        i = 1
        while i < len(elem):
            total += power_rel[harmonics_indexes[elem[i]]]
            i+=1
        rel_power_sums = np.append(rel_power_sums, total)

    relevant_index = peak_indexes[np.argmax(rel_power_sums)]
    relevant_freq = freq[relevant_index]*constant
    #**remember** constant = freq_fact*conv_hr_day
    curr_relevant_freq_values = 1. / np.arange(2,len(original_peak_indexes)) * relevant_freq
    i =0
    while i < len(original_peak_indexes):
        temp_index = original_peak_indexes[i]
        temp_freq = freq[temp_index]*constant
        if temp_freq >= relevant_freq:
            break
        for elem in curr_relevant_freq_values:
            if np.abs(elem - temp_freq) <= error:
                relevant_index = temp_index
                break
        i+=1
            
    relevant_freq = freq[relevant_index]*constant
    relevant_period = 24./relevant_freq

################################################################################
############################ SAVING THE FIGURE #################################
################################################################################

    fig = plt.figure(figsize=(20,15))
    ax1 = fig.add_subplot(211)
    ax1.scatter(t,f, s= 10)
    ax1.plot(t,f, 'black', linewidth = .75)
    plt.title("Lightcurve " + str(chosenfile), fontsize = 16) #or filename
    plt.xlabel("Time (Hours)")
    plt.ylabel("Numerical Flux")
    plt.xlim([np.amin(t),np.amax(t)])
    delta = np.amax(f) - np.amin(f)
    plt.ylim([1 - 1.5*delta, 1. + .5*delta])
    
    ax2 = fig.add_subplot(223)
    ax2.plot(freq*freq_fact*conv_hr_day, power,'black')
    plt.title("Numfreq = " + str(len(peak_indexes)), fontsize =16)
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("Amplitude")
    plt.xlim([0,24])
    
    ax3 = fig.add_subplot(224)
    ax3.plot(freq*freq_fact*conv_hr_day, power_rel,'black')
    ax3.scatter(freq[relevant_index]*freq_fact*conv_hr_day, power_rel[relevant_index], c='black', s=40)
    plt.title("PEAK FREQ = " + str(relevant_freq) + " Period: " + str(relevant_period), fontsize =16)
    #plt.title(char("PEAK FREQ = " + str(relevant_freq),"Period: " + str(relevant_period)), fontsize = 16)
    plt.xlabel("Frequency (cycles/day)")
    upper  = np.round(relevant_freq)
    if upper < relevant_freq:
        upper += 1
    plt.xlim([0,upper])
    
    #plt.show()
    
    info = ["LCfluxesepic" + str(filename) + "star00", relevant_freq, relevant_period]
    information_arr.append(info)
    
    #fig.savefig(chosenfile + "_testgraphs.png", dpi = 300)
    
    
for elem in information_arr:
    print(elem)