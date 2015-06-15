import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftfreq
def periodic_func(xmin, xmax, N, y, num):
    """Returns two arrays (xvalues and yvalues) that are repeated by amount num
    xmin: min of xvalue 
    xmax: max of xvalue 
    N: # of data points
    y: the y-values for the x-values
    num: # of times to repeat graph

    >>> a = 0
    >>> b = 5
    >>> N = 5
    >>> num_repeats = 2
    >>> x = np.linspace(a, b, N)
    >>> print(x)
    [ 0.    1.25  2.5   3.75  5.  ]
    >>> y = np.piecewise(x, [x % b != 1.0, x % b <= 1.0], [-1, 1]) #box function
    >>> print(y)
    [ 1. -1. -1. -1.  1.]
    >>> x1, y1 = periodic_func(a, b, N, y, num_repeats)
    >>> print(x1)
    [  0.     1.25   2.5    3.75   5.     5.     6.25   7.5    8.75  10.  ]
    >>> print(y1)
    [ 1. -1. -1. -1.  1.  1. -1. -1. -1.  1.]
    """
    
    d = xmax - xmin 
    x_ = np.linspace(xmin, xmax, N)
    y_ = y
    i = 1
    while i < num:
        x_ = np.append(x_, np.linspace(xmin + i*d, xmax + i*d, N))
        y_ = np.append(y_, y)
        i += 1
    return x_, y_

#Variables
#a: min value of t, b: max value of t
a = 0
b = 100
#N: number of data points
N = 1000.0 #float
#v1: top value for flux, v2: bottom value for flux
v1 = 1
v2 = .9
num_repeat = 50 #number of times for the base graph to be repeated
d = 5.0 #float #duration of transit
t = np.linspace(a, b, N) #the time values

y_box = np.piecewise(t, [t % b != d, t % b <= d], [v1, v2]) #y-values for box function
    
m1 = 2/d*(v2-v1) #downward slope for triangle 
m2 = m1 * (-1) #upward slope for triangle
y_triangle = np.piecewise(t, [t <= d / 2, t > d /2, t >= d], [lambda x: v1 + m1*x, lambda x: v2 +m2*(x-d/2), v1]) #y-values for triangle function

f = .01
y_sin = np.sin(2*np.pi*f*t) # y-values for sin function

#plot
x1, y1 = periodic_func(a, b, N, y_triangle, num_repeat)
plt.plot(x1, y1)
plt.xlim([a, b*num_repeat])
val_avg = np.average([np.abs(v1), np.abs(v2)])
plt.ylim([v2 - val_avg, v1 + val_avg])
plt.show()



        

        
        
        
