import numpy as np
import matplotlib.pyplot as plt

v1 = 1
v2 = .9
a = -5
b = 5

x = np.linspace(a,b,1000)
y = np.around(x) #round the array
z = np.piecewise(y, [y%2==0, y%2==1], [v1, v2])
plt.plot(x,z)
plt.ylim([0, 2])
plt.show()

#print(x)
#print(y)
#print(z)

