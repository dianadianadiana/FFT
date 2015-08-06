from fft_functions import *
import time

#Variables
extra_fact = 5
x = [peak_constraint, harmonics_constraint, lower_freq, upper_freq, lowest_freq] = [4.0, 3.0, .22, 12.0, .1]
bools = [huge_gap, has_peaks, good_peak, longer_period] = [False, False, False, False]
################################################################################
################################################################################
temppath = "C:/Users/dianadianadiana/Desktop/temp/LC/"
path3 = "C:/Users/dianadianadiana/Desktop/Research/Field3/Decorrelatedphotometry2/Decorrelatedphotometry2/" # 16,339 Files
figurepath = "C:/Users/dianadianadiana/Desktop/trash/"
path = path3

start_time = time.clock() #start the clock

count = 0
filename_arr = read_files(path)
for filename in filename_arr[:15]:
    
    # Part I - read in the file
    dataLC = read_data(path, filename)
    epicname = filename[12:21] # EPIC number
    
    # Part II - fill in the gaps of the LC
    time_cad, flux_cad, huge_gap = gap_filler(dataLC)
    
    # Part III - FFT and FFT Normalization
    freq, power_rel, power, n = fft_part(time_cad, flux_cad, extra_fact)
    
    # Part IV - find the indexes of the peaks
    inds = peak_indexes, harmonics_indexes, original_peak_indexes, original_harmonics_indexes = peak_finder(x, freq, power_rel)
    
    # Part V - find the relevant index,freq,period
    relevant_index, relevant_freq, relevant_period, bools[1:] = find_freq(inds, n, freq, power_rel)
    
    # Part VI - return (and save) the figure
    fig = get_figure(time_cad, flux_cad, bools, inds, x, freq, power_rel, power, n, relevant_index, relevant_freq, relevant_period, epicname)
    fig.savefig(figurepath + str(epicname) + "_figure.png", dpi = 200)
    
    # Part VII - return information about the data
    info = get_info(bools, inds, epicname, relevant_freq, relevant_period)
    print(info)
    count+=1

stop_time = time.clock() - start_time
print stop_time, "seconds"
print "time per LC:", stop_time/count, "seconds"