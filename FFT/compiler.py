from fft_functions import *
import time

#Variables
extra_fact = 5
cut_peak = 3.5
knot1, knot2 = 60,120
x = [peak_constraint, harmonics_constraint, lower_freq, upper_freq, lowest_freq] = [4.0, 3.0, 1, 12.0, 1]
bools = [huge_gap, has_peaks, good_peak, longer_period] = [False, False, False, False]
################################################################################
################################################################################
## different locations where data is coming from
temppath = "C:/Users/dianadianadiana/Desktop/temp/LC/"
path3 = "C:/Users/dianadianadiana/Desktop/Research/Field3/Decorrelatedphotometry2/Decorrelatedphotometry2/" # 16,339 Files
datapath = "C:/Users/dianadianadiana/Desktop/Research/Data/LC/"

#EPIC3 = array of the EPIC numbers of the planet candidates from K2 Field 3
EPIC3 = ['206011496','206038483','206245553','206247743','205924614','206026904','206036749',
'206044803','206061524','206096602','206114294','206155547','206169375','206181769',
'206298289','206318379','206432863','206500801','205947161','206011691','206159027']

path = path3
figurepath = "C:/Users/dianadianadiana/Desktop/TRASH1/" #destination of where to save graph
information_path = figurepath + "information.txt" #saves all the information for the files
interest_path = figurepath + "interest.txt" #saves the information for the files of interest 

information_file = open(information_path, 'w')
interest_file = open(interest_path, 'w')

count = 0
filename_arr = read_files(path)
start_time1 = time.clock()
for index, filenameLC in enumerate(filename_arr[:], start = 0):
    #if iterating through epicnames..
#for index, epicname in enumerate(EPIC3[:1], start = 0):
    #filenameLC = "LCfluxesepic" +str(epicname) + "star00.txt"
    
    print "index", index, "count", count
    start_time = time.clock() #start the clock
    
    # Part I - read in the file
    dataLC = read_data(path+filenameLC) # dataLC[0] = time, dataLC[1] = flux
    epicname = filenameLC[12:21] # EPIC number
    time1 = time.clock() - start_time
    print "time1", time1
    
    # Part II - fill in the gaps of the LC
    time_cad, flux_cad, huge_gap = gap_filler(dataLC)
    time2 = time.clock() - time1 - start_time
    print "time2", time2
    
    # Part III - FFT and FFT Normalization
    freq, power, n, bin_sz, peak_width = fft_part(time_cad, flux_cad, extra_fact)
    power_rel = fft_normalize(freq, power, n, bin_sz, cut_peak, knot1, knot2)
    time3 = time.clock() - time2 - start_time
    print "time3", time3
    
    # Part IV - find the indexes of the peaks
    inds = peak_indexes, harmonics_indexes, original_peak_indexes, original_harmonics_indexes = peak_finder(x, freq, power_rel)
    time4 = time.clock() - time3 - start_time
    print "time4", time4
    
    # Part V - find the relevant index,freq,period
    relevant_index, relevant_freq, relevant_period, bools[1:], potential_arr, rel_power_sums  = find_freq(inds, n, freq, power_rel)
    huge_gap, has_peaks, good_peak, longer_period = bools
    time5 = time.clock() - time4 - start_time
    print "time5", time5
    
    # Part VI - return (and save) the figure
    fig = get_figure(time_cad, flux_cad, bools, inds, x, freq, power_rel, power, n, relevant_index, relevant_freq, relevant_period, epicname)
    time6 = time.clock() - time5- start_time
    print "time6", time6
    fig.savefig(figurepath + str(epicname) + "_50.png", dpi = 50)
    time7 = time.clock() - time6- start_time
    print "time7", time7
    
    #Part VII - return information about the data
    info = get_info(bools, inds, epicname, relevant_freq, relevant_period)
    print info
    information_file.write("%s\n" % info)
    if info[1] != 1:
        interest_file.write("%s\n" % epicname)
    
    stop_time = time.clock() - start_time
    print stop_time, "seconds"
    count+=1

stop_time1 = time.clock() - start_time1
print "Total time:", stop_time1, "seconds"
print "Time per LC:", stop_time1/count, "seconds"

information_file.close()
interest_file.close()