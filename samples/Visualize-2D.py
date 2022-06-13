import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def import_json(file_name):
    f = open(file_name)
    result = json.load(f)
    return result

def plot_result(title, data):
    fig, axis = plt.subplots(1, 1)

    plt.title(title)

    axis.scatter(np.matrix(data['train_set']['inputs']['x']), np.matrix(data['train_set']['inputs']['y']), np.matrix(data['train_set']['outputs']), color='green', marker='o')

    pred_xy_grid['x'] = np.matrix(data['predictions']['inputs']['x'])
    pred_xy_grid['y'] = np.matrix(data['predictions']['inputs']['y'])

    axis.plot_surface(pred_xy_grid['x'], pred_xy_grid['y'], np.matrix(data['predictions']['expected']), color='red')

    pred_mean = np.matrix(data['predictions']['means'])
    axis.plot_surface(pred_xy_grid['x'], pred_xy_grid['y'], pred_mean, color='blue', alpha=0.5)

    pred_sigmas = np.matrix(data['predictions']['sigmas'])
    axis.plot_surface(pred_xy_grid['x'], pred_xy_grid['y'], pred_mean - 2.5 * pred_sigmas, color='blue', alpha=0.3)
    axis.plot_surface(pred_xy_grid['x'], pred_xy_grid['y'], pred_mean + 2.5 * pred_sigmas, color='blue', alpha=0.3)

    # axis.legend()

file_name = 'Log.json'
if(1 < len(sys.argv)):
    file_name = sys.argv[1]

title_passed = ''
if(2 < len(sys.argv)):
    title_passed = sys.argv[2]

data = import_json(file_name)
if len(title_passed) == 0:
    for title in data:
        plot_result(title, data[title])
else:
    plot_result(title_passed, data[title_passed])
plt.show()











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
