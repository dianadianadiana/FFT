import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np

################################################################################
################################################################################
############################ SAVING THE FIGURE #################################
################################################################################
################################################################################

def plot_figure(time_cad, flux_cad, freq, power, power_rel, peak_indexes):
    fig = plt.figure(figsize=(20,15))
    
    ax1 = fig.add_subplot(211)
    time_cad /= 24
    ax1.scatter(time_cad, flux_cad, s=10, c='black')
    ax1.plot(time_cad, flux_cad, 'black', linewidth = .75)
    ax1.set_title("Lightcurve " + str(epicname), fontsize = 16)
    ax1.set_xlabel("Time (Days)")
    ax1.set_ylabel("Numerical Flux")
    ax1.set_xlim([np.amin(time_cad), np.amax(time_cad)])
    delta = np.amax(flux_cad) - np.amin(flux_cad)
    ax1.set_ylim([1 - 1.5 * delta, 1. + .5 * delta])
    
    ax2 = fig.add_subplot(223)
    ax2.plot(freq*constant, power, 'black')
    if has_peaks:
        ax2.scatter(freq[peak_indexes]*constant, power[peak_indexes], s=30, c="black")
        ax2.set_title("Numfreq = " + str(len(peak_indexes)), fontsize = 16)
    else:
        ax2.set_title("NO PEAKS DETECTED")
    ax2.set_xlabel("Frequency (cycles/day)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim([0,10])
    ax2.set_ylim(bottom=0)
    
    ax3 = fig.add_subplot(224)
    ax3.plot(freq*constant, power_rel,'black')
    if has_peaks and good_peak:
        ax3.scatter(freq[relevant_index]*constant, power_rel[relevant_index], c='black', s=50)
        ax3.set_title("PEAK FREQ = " + str(relevant_freq) + " Period: " + str(relevant_period), fontsize =16)
    else:
        ax3.set_title("NO PEAKS DETECTED")
    ax3.set_xlabel("Frequency (cycles/day)")
    ax3.set_ylabel("Relative Amplitude")
    upper = 10
    ax3.set_xlim([0, upper])
    ax3.set_ylim([0, 1.5*np.amax(power_rel[n:])])
    
    plt.show()
    
    ############################ Saving the Figure #############################
    figurepath = "C:/Users/dianadianadiana/Desktop/temp/Figures/"
    epicname_no_txt = epicname[:len(epicname)-4]
    fig.savefig(figurepath + str(epicname_no_txt) + "_figure.png", dpi = 200)

