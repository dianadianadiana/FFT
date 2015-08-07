from fft_functions import *
import time

#Variables
extra_fact = 5
cut_peak = 2.5
knot1, knot2 = 60,120
x = [peak_constraint, harmonics_constraint, lower_freq, upper_freq, lowest_freq] = [4.0, 3.0, .22, 12.0, .1]
bools = [huge_gap, has_peaks, good_peak, longer_period] = [False, False, False, False]
################################################################################
################################################################################
temppath = "C:/Users/dianadianadiana/Desktop/temp/LC/"
path3 = "C:/Users/dianadianadiana/Desktop/Research/Field3/Decorrelatedphotometry2/Decorrelatedphotometry2/" # 16,339 Files
figurepath = "C:/Users/dianadianadiana/Desktop/trash/FIELD3(1)/"
datapath = "C:/Users/dianadianadiana/Desktop/Research/Data/LC/"
path = path3

#start_time = time.clock() #start the clock

count = 0
filename_arr = read_files(path)
EPIC3 = ['206011496','206038483','206245553','206247743','205924614','206026904','206026904','206036749',
'206044803','206061524','206096602','206096602','206114294','206155547','206169375','206181769',
'206298289','206318379','206432863','206500801','205947161','206011691','206159027']
for filename in filename_arr[100:200]:
#for epicname in EPIC3:
    start_time = time.clock() #start the clock
    print start_time
    #filename = "LCfluxesepic" +str(epicname) + "star00.txt"
    # Part I - read in the file
    dataLC = read_data(path, filename)
    epicname = filename[12:21] # EPIC number
    time1 = time.clock() - start_time
    print "time1", time1
    # Part II - fill in the gaps of the LC
    time_cad, flux_cad, huge_gap = gap_filler(dataLC)
    time2 = time.clock() - time1 - start_time
    print "time2", time2
    # Part III - FFT and FFT Normalization
    freq, power_rel, power, n = fft_part(time_cad, flux_cad, extra_fact, cut_peak, knot1, knot2)
    time3 = time.clock() - time2- start_time
    print "time3", time3
    # Part IV - find the indexes of the peaks
    inds = peak_indexes, harmonics_indexes, original_peak_indexes, original_harmonics_indexes = peak_finder(x, freq, power_rel)
    time4 = time.clock() - time3- start_time
    print "time4", time4
    # Part V - find the relevant index,freq,period
    relevant_index, relevant_freq, relevant_period, bools[1:] = find_freq(inds, n, freq, power_rel)
    time5 = time.clock() - time4- start_time
    print "time5", time5
    # Part VI - return (and save) the figure
    fig = get_figure(time_cad, flux_cad, bools, inds, x, freq, power_rel, power, n, relevant_index, relevant_freq, relevant_period, epicname)
    time6 = time.clock() - time5- start_time
    print "time6", time6
    fig.savefig(figurepath + str(epicname) + "_figure7.80.png", dpi = 80)
    time7 = time.clock() - time6- start_time
    print "time7", time7
    # Part VII - return information about the data
    info = get_info(bools, inds, epicname, relevant_freq, relevant_period)
    print(info)
    time8 = time.clock() - time7- start_time
    print "time8", time8
    count+=1
    print(count)
    stop_time = time.clock() - start_time
    print stop_time, "seconds"

stop_time = time.clock() - start_time
print stop_time, "seconds"
print "time per LC:", stop_time/count, "seconds"