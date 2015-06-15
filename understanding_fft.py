from scipy.fftpack import fft,fftfreq
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi,sin,cos

### http://bender.astro.sunysb.edu/classes/phy688_spring2013/lectures/ffts.pdf
### focus starting on pg 17

f = 5.1 #frequency
N = 500.0 #float #Number of data points
a = 0
b = 50
d = (b-a)/N #distance between points
x = np.linspace(a,b,N,endpoint = False)

nyquistfreq = 0.5/d

print("Nyquist frequency: " + str(nyquistfreq))

#https://en.wikipedia.org/wiki/Nyquist_frequency

y_sin = sin(2*pi*f*x)
y_cos = cos(2*pi*f*x)
y_sin_shift = sin(2*pi*f*x + pi/ 4)

##Question: Why is the amplitude not 0 - 1?
#Answer http://matteolandi.blogspot.com/2010/08/notes-about-fft-normalization.html
normfact = 2.0/N
fy_sin = fft(y_sin)*normfact# * (f/(N*d))
fy_sin_real = np.real(fy_sin)
fy_sin_imag = np.imag(fy_sin)
#power = np.abs(fy_sin)**2
freqs = fftfreq(np.int(N))
freqs_shift = np.fft.fftshift(freqs)
#plt.plot(x, fy_sin)
#plt.plot(x, power)

#plot

freqfact = 1.0/d

## Question: What is the purpose of fftshift? Nyquist limit?
plt.plot(freqs*freqfact, fy_sin_real, 'g')
plt.plot(freqs*freqfact, fy_sin_imag, 'g--')
plt.plot(freqs_shift*freqfact, fy_sin_real, 'b')
plt.plot(freqs_shift*freqfact, fy_sin_imag, 'b--')

pos_max_power =np.argmax(fy_sin_real)
print("frequency of max power: " + str(freqs[pos_max_power]*freqfact))
print("frequency of max power shifted: " + str(freqs_shift[pos_max_power]*freqfact))


#plt.plot(freqs, power)
plt.title('Solid = Real, Dashed = Imaginary\nGreen = No Shift, Blue = Shift')
plt.xlabel('Frequency')
plt.ylabel('FFT')
plt.show()
plt.close()


