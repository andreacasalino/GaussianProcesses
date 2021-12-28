import numpy as np
import matplotlib.pyplot as plt
import json


f = open('predictions.json')
data1 = json.load(f)
f = open('predictions_after_tune.json')
data2 = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2)

def plot_data(data_, axis):
    input_predictions = np.array(data_['input_predictions'])
    prediction_covariances = np.array(data_['prediction_covariances'])
    prediction_means = np.array(data_['prediction_means'])

    axis.plot(input_predictions, data_['output_predictions'], color='blue', label='real function')
    axis.plot(input_predictions, prediction_means, color='red', label='GP predictions')
    axis.scatter(data_['input_samples'], data_['output_samples'], color='red', marker='o', label="train set")

ax1.set_title('Predictions un-tuned hyperparameters')
plot_data(data1, ax1)
ax1.legend()

ax2.set_title('Predictions tunining the hyperparameters')
plot_data(data2, ax2)
ax2.legend()

plt.show()
