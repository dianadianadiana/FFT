import numpy as np
import matplotlib.pyplot as plt

#variables:
#v1: the higher value; v2: the lower value
v1 = 1
v2 = .9
#delta1: the error for v1; delta2: the error for v2
delta1 = .01 
delta2 = .005
#[v1_high,v1_low): range for v1; [v2_high,v2_low): range for v2
v1_high = v1 + delta1
v1_low = v1 - delta1
v2_high = v2 + delta2
v2_low = v2 - delta2
#[xmin, xmax] for the graph
xmin = -5
xmax = 5
#N: segments/# of data points
N = 100

#arrays
x = np.linspace(xmin, xmax, N) #x-values
z = np.around(x) #round the array
y = np.empty(len(x)) #create an empty array w/ same len as x -- will be y-values

#Note: to sample random float in [a,b) --> (b - a) * np.random.random_sample() + a
for i in np.arange(len(x)):
    if z[i] % 2 == 0: 
        y[i] = (v1_high - v1_low) * np.random.random_sample() + v1_low
    else:
        y[i] = (v2_high - v2_low) * np.random.random_sample() + v2_low
        
#plot
plt.plot(x,y)
plt.ylim([0, 1.5])
plt.xlim([xmin, xmax])
plt.title("Random Box Function")
plt.grid()
plt.show()

#Tests
#print(x)
#print(z)
#print(y)

