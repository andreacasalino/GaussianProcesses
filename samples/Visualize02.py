import numpy as np
import matplotlib.pyplot as plt
import json


f = open('predictions.json')
data = json.load(f)

fig, ax = plt.subplots(1, 1)
ax.plot(data['input_predictions'], data['output_predictions'], color='blue')
ax.plot(data['input_predictions'], data['prediction_means'], color='red')
ax.plot(data['input_samples'], data['output_samples'], color='red', marker='o')
plt.show()
