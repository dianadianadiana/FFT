import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftfreq

#variables:
#v1: the higher value; v2: the lower value
v1 = 1
v2 = .9
#[xmin, xmax] for xvalues of the graph
xmin = 0
xmax = 84
#N: segments/# of data points
#smaller N --> more triangular (100)
#bigger N --> more boxy (1000)
#middle N --> more trapezoidal (300)
N = 10000.0

#period: the planet's period around the star in hours
period = 8.3
d = (xmax -xmin)/N #distance between points

#arrays
x = np.linspace(xmin, xmax, N) #x-values (time)
z = np.around(x) #round the array
#y = np.piecewise(z, [z % period != 0.0, z % period == 0.0], [v1, v2])
y = np.piecewise(x, [x % period > 1.0, x % period <= 1.0], [v1, v2]) #good for box
#y = np.piecewise(z, [z % period != 0.0, z % period == 0.0], [v1, v2])
#z1 = z / period
#z2 = z % period
#print(x)
#print(z)
#print(z1)
#print(z2)

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

#plot FFT
plt.figure()

norm_fact = 2.0/N
fy = fft(y) * norm_fact
fy_real = np.real(fy)
fy_imag = np.imag(fy)

freq = fftfreq(np.int(N))
freq_fact = 1.0 / d
print(freq_fact)

plt.plot(freq *freq_fact, np.abs(fy))
plt.plot(freq*freq_fact, fy_real)

plt.title("FFT")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()

pos_max_power = np.argmax(fy_real)
print("frequency of max power: " + str(freq[pos_max_power]*freq_fact))
