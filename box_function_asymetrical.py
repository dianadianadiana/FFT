import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftfreq

#variables:
#v1: the higher value; v2: the lower value
v1 = 1
v2 = .9
#[xmin, xmax] for xvalues of the graph
xmin = 0
xmax = 100
#N: segments/# of data points
#smaller N --> more triangular
#bigger N --> more boxy
N = 1000
#period: the planet's period around the star
period = 8.5

#arrays
x = np.linspace(xmin, xmax, N) #x-values (time)
z = np.around(x) #round the array
y = np.piecewise(z, [z%period!=0, z%period==0], [v1, v2])

#plot

plt.figure()
v_avg = np.average([np.abs(v1), np.abs(v2)])
plt.ylim([v2 - v_avg, v1 + v_avg])
plt.xlim([xmin, xmax])
plt.title("Periodic Function")
plt.xlabel("Time (hours)")
plt.ylabel("Flux")
#plt.grid()
plt.plot(x,y)
#plt.plot(x,y)#, 'b+')

#plot FFT
plt.figure()
fy = fft(y)
freq = fftfreq(np.int(N)) 
plt.title("FFT")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.plot(freq,np.abs(fy))
plt.show()