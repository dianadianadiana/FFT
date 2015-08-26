import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from pylab import *
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
    
def constraint_index_finder(constraint, y):
    """ 
    Parameters:
    constraint: the number times mean of y
    y: y values -- power
    Return value: 
    val_indexes -- array with indexes where y[indexes] output values greater than constraint * mean value
    What it does:
    The mean of the array is multiplied with the constraint, and the index of any value in y (power) 
    that is above constraint*mean value is stored to an array --> That array is then returned
    """
    return np.where(y >= constraint*np.median(y))[0]
    
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
############################# read all the data ################################
################################################################################
################################################################################

information_arr =[] # each elem has this format [ filename, code #, peak freq, period ]

path1 = "C:/Users/dianadianadiana/Desktop/Decorrelatedphotometry2/Decorrelatedphotometry2/"
path2 = "C:/Users/dianadianadiana/Desktop/Research/Field2/Decorrelatedphotometry2/"
path3 = "C:/Users/dianadianadiana/Desktop/Research/Field3/Decorrelatedphotometry2/Decorrelatedphotometry2/" # 16,339 Files
datapath = "C:/Users/dianadianadiana/Desktop/Research/Data/LC/" # 16 files
temppath = "C:/Users/dianadianadiana/Desktop/temp/LC/"
beginning = "LCfluxesepic"
#ending = "star00"

path = path3
filename_arr = os.listdir(path)

#all the field 3 interest 
interest_epic_arr = ["205906302","205914832","205924614","205945953","205962680","205977782", "205982900",
"205985357","205988562","205992585","206038285","206038483","206049101","206103150","206135267",
"206143957","206152015","206154641","206159239","206202136","206259533","206260730","206311743",
"206409426","206412289","206474395","206477939","206489474","206500801","206511754","206532093",
"205906302","205962262","205968100","205978103"]
#the potential planets of field 3
planets_epic_arr = ['205906302','205924614', '205945953', '205962262', '205962680',
'205985357','205992585','206038285', '206038483', '206049101', '206103150', '206135267',
'206202136', '206311743', '206500801']

interest_wo_planets_arr = ["205914832","205977782", "205982900","205988562","206143957",
"206152015","206154641","206159239","206259533","206260730","206409426","206412289",
"206474395","206477939","206489474","206511754","206532093",
"205906302","205962262","205968100","205978103"]

EPIC = ['206011496','206038483','206245553','206247743','205924614','206026904','206026904','206036749',
'206044803','206061524','206096602','206096602','206114294','206155547','206169375','206181769',
'206298289','206318379','206432863','206500801','205947161','206011691','206159027']
#chosenfile = "C:/Users/dianadianadiana/Desktop/Research/Field3/information.txt"
#data = np.genfromtxt(chosenfile, delimiter=',', dtype=None, names=('epic','code','freq','period'))
#keep = []
#for elem in data:
#    if elem[2] != 0:
#        keep.append(elem)
        
count = 0
#for epicname in keep[0:10]:
#for epicname in filename_arr[100:500]:
for epicname in EPIC[:10]:
    count+=1
    epicname = beginning + str(epicname) + "star00.txt"
    #bad data files from field 3
    if epicname == "LCfluxesepic206134600star00.txt" or epicname == "LCfluxesepic206050032star00.txt":
        continue
    chosenfile = path + str(epicname)
    #print(chosenfile)
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
    gap = gap.astype(int)
    
    gap_cut = 1.1   #Time differences greater than gap_cut*cadence are gaps    
    gap_loc = np.where(gap > gap_cut)[0]
    num_gap = len(gap_loc) # the number of gaps present 
    biggest_gap = np.amax(gap)

################################################################################
########## Create a new time and flux array to account for the gaps ############
################################################################################

    num_cad = sum(gap)  # the number of data points
    time_cad = np.arange(num_cad)*cadence + np.amin(t)
    flux_cad = np.empty(num_cad) 
        
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
    d_pts = (np.amax(time_cad) - np.amin(time_cad)) / (N-1)
    freq_fact = 1.0 / d_pts #frequency factor 
    freq *= freq_fact
    
    postivefreq = freq > 0 # take only positive values
    freq, f_flux = freq[postivefreq], f_flux[postivefreq]
    
    power = np.abs(f_flux)#**2
    
    bin_sz = 1./len(newf) * freq_fact # distance between consecutive points in cycles per day
    peak_width = 2 * bin_sz * 2**extra_fact #in cycles per day

################################################################################
########################### Normalizing the FFT ################################
################################################################################

    knot_w = 60*n*bin_sz # difference in freqs between each knot
    
    ## first fit
    first_knot_i = np.round(knot_w/bin_sz) #index of the first knot is the first point that is knot_w away from the first value of x 
    last_knot_i = len(freq) - first_knot_i#index of the last knot is the first point that is knot_w away from the last value of x
    knots = np.arange(freq[first_knot_i], freq[last_knot_i],knot_w)
    spline = LSQUnivariateSpline(freq, power, knots, k=2) #the spline, it returns the piecewise function
    fit = spline(freq) #the actual y values of the fit
    if np.amin(fit) < 0:
        fit = np.ones(len(freq))
        print "fit"
    pre_power_rel = power/fit
    
    # second fit -- by deleting the points higher than 4 times the average value of pre_power_rel
    #pre_indexes = constraint_index_finder(4**2, pre_power_rel) #peaks
    pre_indexes = constraint_index_finder(8, power)
    #power_fit = np.delete(power, pre_indexes)
    power_fit = np.delete(pre_power_rel, pre_indexes)
    #SHOULD POWER FIT BE W POWER OR PRE POWER REL
    freq_fit = np.delete(freq, pre_indexes)
    
    knot_w1 = 120 * n *bin_sz
    first_knot_fit_i = np.round(knot_w1/bin_sz) #index of the first knot is the first point that is knot_w away from the first value of x 
    last_knot_fit_i = len(freq_fit) - first_knot_fit_i#index of the last knot is the first point that is knot_w away from the last value of x
    knots_fit = np.arange(freq_fit[first_knot_fit_i], freq_fit[last_knot_fit_i], knot_w1)
    spline = LSQUnivariateSpline(freq_fit,power_fit,knots_fit, k=2) #the spline, it returns the piecewise function
    fit3 = spline(freq) #the actual y values of the fit applied to freq
    if np.amin(fit3) < 0:
        fit3 = np.ones(len(freq))
        print "fit3"
    
    # relative power
    #power_rel = power / fit3 #* 10**5
    
    power_rel = pre_power_rel / fit3
    power_rel /= np.median(power_rel)


################################################################################
################################################################################
############################### Peak Limits ####################################
################################################################################   
################################################################################

    peak_constraint, harmonics_constraint = 3.5, 3

    val_indexes = constraint_index_finder(peak_constraint, power_rel) #peaks
    val_indexes1 = constraint_index_finder(harmonics_constraint, power_rel) #harmonics

    peak_indexes = max_peaks(val_indexes, power_rel)
    harmonics_indexes = max_peaks(val_indexes1, power_rel)
    
    # keep all of the original peak_indexes/harmonics_indexes to check later on 
    # if it's a longer period planet
    highest_period = 105.0 # in hours
    lowest_freq = 24. / highest_period # in cycles per day
    original_peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes],.1))
    original_harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes],lowest_freq))
    
    # we only want peaks that are between freqs of [1,10] cycles/day
    lower_freq, upper_freq = 24. / highest_period, 12.0
    peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes],lower_freq,upper_freq))
    harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes],lower_freq,upper_freq))
    
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
        
        relevant_freq = freq[relevant_index]
        #relevant_period = 24./relevant_freq
        relevant_period = relevant_freq**(-1)

################################################################################
################################################################################
############################ SAVING THE FIGURE #################################
################################################################################
################################################################################

    fig = plt.figure(figsize=(20,15))
    
    ax1 = fig.add_subplot(221)
    ax1.scatter(time_cad, flux_cad, s=5, c='black')
    ax1.plot(time_cad, flux_cad, 'black', linewidth = .75)
    ax1.set_title("Lightcurve " + str(epicname), fontsize = 16)
    ax1.set_xlabel("Time (Days)")
    ax1.set_ylabel("Numerical Flux")
    ax1.set_xticks(np.arange(np.round(np.amin(time_cad),-1), np.round(np.amax(time_cad),-1)+1, 5.0))
    ax1.set_xlim([np.amin(time_cad), np.amax(time_cad)])
    delta = np.amax(flux_cad) - np.amin(flux_cad)
    ax1.set_ylim([1 - 1.5 * delta, 1. + .5 * delta])
    ax1.grid(True)
    
    lower = 0
    upper = 12
    ax2 = fig.add_subplot(223)
    ax2.plot(freq, power, 'black',linewidth = .75)
    if has_peaks:
        ax2.scatter(freq[peak_indexes], power[peak_indexes], s=30, c="black")
        #ax2.scatter(freq[harmonics_indexes], power[harmonics_indexes], s=20, c="blue")
        ax2.set_title("Numfreq = " + str(len(peak_indexes)), fontsize = 16)
    else:
        ax2.set_title("NO PEAKS DETECTED")
    ax2.set_xlabel("Frequency (cycles/day)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim([lower,upper])
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(np.arange(upper))
    ax2.grid(True)
    
    ax3 = fig.add_subplot(224)
    ax3.plot(freq, power_rel,'black',linewidth = .75)
    if has_peaks and good_peak:
        ax3.scatter(freq[relevant_index], power_rel[relevant_index], c='black', s=50)
        ax3.set_title("PEAK FREQ = " + str(relevant_freq) + " Period: " + str(relevant_period) + " days", fontsize =13)
        #ax3.scatter(freq[original_peak_indexes], power_rel[original_peak_indexes], s=30, c="black")

    else:
        ax3.set_title("NO PEAKS DETECTED")
    ax3.set_xlabel("Frequency (cycles/day)")
    ax3.set_ylabel("Relative Amplitude")
    ax3.set_xlim([lower, upper])
    ax3.set_ylim([0, 1.5*np.amax(power_rel[n:])])
    ax3.set_xticks(np.arange(upper))
    ax3.grid(True)

    ### Folded Light Curve
    ax4 = fig.add_subplot(222)
    ax4.set_xlabel("Orbital phase")
    ax4.set_ylabel("Relative Flux")
    ax4.grid(True)
    if has_peaks and good_peak:
        phases, orbit = np.modf(time_cad/relevant_period)
        ax4.plot(phases,flux_cad,'k.')
        ax4.set_title("Folded light curve")
    else:
        ax4.set_title("Folded light curve - no peaks detected")

    
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
    print(info)
    
    epicname_no_txt = epicname[:len(epicname)-4]
    figurepath = "C:/Users/dianadianadiana/Desktop/Research/Field3/Figures8.5.15/"
    #figurepath = "C:/Users/dianadianadiana/Desktop/Research/Data/Figures/Figures_Spline/"
    #figurepath = "C:/Users/dianadianadiana/Desktop/temp/Figures/"
    #figurepath = "C:/Users/dianadianadiana/Desktop/Research/Field2/Figures/"
    #figurepath = "C:/Users/dianadianadiana/Desktop/trash/"

    #fig.savefig(figurepath + str(epicname_no_txt) + "_figure.png", dpi = 200)
    
    #plt.close()
    
    if count%10 == 0:
        print(count)
    #print (relevant_period - 1.7292)/1.7292

################################################################################   
############################ Saving the Information ############################
################################################################################
#http://gis.stackexchange.com/questions/72458/export-list-of-values-into-csv-or-txt-file

#res = information_arr
#csvfile = "C:/Users/dianadianadiana/Desktop/Research/Field3/Interest/Interest_but_not_potential_planets/information_period_in_days.txt"
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
#    
stop_time = time.clock() - start_time
print stop_time, "seconds"
print "time per LC:", stop_time/count, "seconds"
