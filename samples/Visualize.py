import numpy as np
import matplotlib.pyplot as plt

matrix = np.loadtxt('predictions_1d.txt')

fig, ax = plt.subplots(1, 1)

ax.plot(matrix[:,0], matrix[:,1], color='red')

ax.plot(matrix[:,0], matrix[:,2], color='green')
ax.plot(matrix[:,0], matrix[:,3], color='green')
ax.plot(matrix[:,0], matrix[:,4], color='green')

plt.show()