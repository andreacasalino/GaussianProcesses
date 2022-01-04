import numpy as np
import matplotlib.pyplot as plt


evolution = np.loadtxt('tune_evolution.txt')

fig, ax = plt.subplots(1, 1)
ax.set_title('Logarithmic likelihhod evolution od the model')
ax.plot(evolution[:,0], evolution[:,1])

plt.show()
