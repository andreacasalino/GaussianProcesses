import numpy as np
import matplotlib.pyplot as plt
import json


f = open('predictions.json')
data = json.load(f)

fig = plt.figure(figsize=plt.figaspect(0.5))

def plot_samples(data_, axis):
    input_x_coord_predictions = np.matrix(data_['input_x_coord_predictions'])
    input_y_coord_predictions = np.matrix(data_['input_y_coord_predictions'])
    output_predictions = np.matrix(data_['output_predictions'])

    axis.plot_surface(input_x_coord_predictions, input_y_coord_predictions, output_predictions, color='blue', alpha=0.5)
    axis.scatter(data_['input_x_samples'], data_['input_y_samples'], data_['output_samples'], color='green', marker='o')

def plot_predictions(data_, axis):
    input_x_coord_predictions = np.matrix(data_['input_x_coord_predictions'])
    input_y_coord_predictions = np.matrix(data_['input_y_coord_predictions'])
    prediction_covariances = np.matrix(data_['prediction_covariances'])
    prediction_means = np.matrix(data_['prediction_means'])
    output_predictions = np.matrix(data_['output_predictions'])

    axis.plot_surface(input_x_coord_predictions, input_y_coord_predictions, output_predictions, color='blue', alpha=0.5)
    axis.plot_surface(input_x_coord_predictions, input_y_coord_predictions, prediction_means, color='red')

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title('Training set points')
plot_samples(data, ax1)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title('Gaussian Process predictions')
plot_predictions(data, ax2)

plt.show()
