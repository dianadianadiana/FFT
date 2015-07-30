import numpy as np

####################### read in data ###############################
temppath = "C:/Users/dianadianadiana/Desktop/temp/LC/"

path =temppath
epicname = "LCfluxesepic201637175star00.txt"
chosenfile= path + str(epicname)
print(chosenfile)
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

def gap_filler(t, f):
    num_time = len(t) # the original number of data time points (w gaps)
        
    delta_t = t[1:num_time]-t[0:num_time-1] # has a length = len(t) - 2 (bc it ignores endpoints)
    cadence = np.median(delta_t) # what the time interval between points with gaps currently is
    #Fill gaps in the time series (according to 2012 thesis method)
    gap = delta_t/cadence # has a length = len(t) - 2 (bc it ignores endpoints)
    gap = np.around(gap)
    gap = np.append(gap, 1)
    gap = gap.astype(int)
        
    gap_cut = 1.1   #Time differences greater than gap_cut*cadence are gaps    
    gap_loc = [i for i in np.arange(len(gap)) if (gap[i] > gap_cut)]
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
    return time_cad, flux_cad, biggest_gap/len(time_cad) >= .2