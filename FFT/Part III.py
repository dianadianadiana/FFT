import numpy as np

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

def max_peaks(index_arr, arr):
    return [index_arr[i] for i in range(1, len(index_arr)-1) if arr[index_arr[i-1]]<arr[index_arr[i]] and arr[index_arr[i+1]]<arr[index_arr[i]]]
    
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
############################### Peak Limits ####################################
################################################################################   
################################################################################

peak_constraint, harmonics_constraint = 4.6, 3.0
lower_freq, upper_freq = 1.0, 10.0
###constant = NEED TO DEAL WITH THIS

def peaks(freq, power_rel, constant, peak_constraint = 4.0, harmonics_constraint = 3.0, lower_freq = 1.0, upper_freq = 10.0):
    """
    Parameters:
        freq                 -- the frequency 
        power_rel            --
        peak_constraint      --
        harmonics_constraint --
        lower_freq           --
        upper_freq           --
    """
    
    val_indexes = constraint_index_finder(peak_constraint, freq, power_rel) #peaks
    val_indexes1 = constraint_index_finder(harmonics_constraint, freq, power_rel) #harmonics

    peak_indexes = max_peaks(val_indexes, power_rel)
    harmonics_indexes = max_peaks(val_indexes1, power_rel)
    
    # keep all of the original peak_indexes/harmonics_indexes to check later on 
    # if it's a longer period planet
    highest_period = 180.0 # in hours
    lowest_freq = 24. / highest_period # in cycles per day
    original_peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes]*constant,lowest_freq))
    original_harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes]*constant,lowest_freq))
    
    # we only want peaks that are between freqs of [lower_freq, upper_freq] cycles/day
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
            poss_indexes_lower = poss_indexes - 2*n # lower bound of possible indexes
            poss_indexes_upper = poss_indexes + 2*n # upper bound of possible indexes
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
    return relevant_freq, relevant_period, peak_indexes, harmonics_indexes, has_peaks, good_peak, longer_period