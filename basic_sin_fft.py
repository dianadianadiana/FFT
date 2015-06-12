import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

#Variables
#v1: top value; v2: bottom value
v1 = 1
v2 = 0
#graph x axis is [xmin, xmax]
xmin = -5
xmax = 5
#N: number of data points
N = 500.0 #float
#distance between points
d_btwn_pts = (xmax-xmin)/N
period = .2 #float

t = np.linspace(xmin, xmax, N) #x values (represent the time)
y = np.sin((t*np.pi*2)/period) #+ 3*np.sin((t*np.pi*2)/(2*period))

#When the input a is a time-domain signal and A = fft(a), 
#np.abs(A) is its amplitude spectrum and np.abs(A)**2 is its power spectrum. 
#The phase spectrum is obtained by np.angle(A).

Y = fft(y) #real and imaginary fft
print('Y: ' + str(Y))
power = np.abs(Y) #do we need to square?
print('power: ' + str(power))

freqs = np.fft.fftfreq(np.int(N)) 
keep = freqs > 0 #keep only pos values
freqs, power = freqs[keep], power[keep]
print(freqs)

#plot
plt.plot(t,y)
plt.plot(freqs,power)
print("max power: " + str(np.amax(power)))
pos_max_power =np.argmax(power)
print("frequency of max power: " + str(freqs[pos_max_power]))
highest_freq = freqs[np.argmax(power)] # highest freq in 1 divided by points
highest_period = 1 / highest_freq 
print("detected period: " + str(highest_period * d_btwn_pts))
print("input period: " + str(period))

plt.grid()
plt.show()

