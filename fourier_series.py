import math
import numpy as np
import matplotlib.pyplot as plt
#Square Wave approximation by Fourier Series:
#y=4/pi*(sin(x) + sin(3x)/3 + sin(5x)/5 + ... sin(nx)/n) terms

n = 100

# generate x's to calculate the points. 
x = np.arange(0,360,1)
#generate the odd numbers to add the sines terms
odd = np.arange(1,n+2,2)
normal = 4/math.pi #just for normalize the sequence to converge to 1 (not pi/4)
#calculate the points

y = reduce(np.add, (np.sin(np.radians(i*x))/i for i in odd)) * normal

#plot
plt.figure()
plt.plot(x,y)
plt.title(str(n)+str(" terms"))
plt.grid()
plt.show()