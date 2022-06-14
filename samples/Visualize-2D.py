import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import sys

def import_json(file_name):
    f = open(file_name)
    result = json.load(f)
    return result

def plot_result(title, data):
    fig = plt.figure()
    plt.title(title)

    samples_xy_grid = {}
    samples_xy_grid['x'] = np.matrix(data['train_set']['inputs']['x'])
    samples_xy_grid['y'] = np.matrix(data['train_set']['inputs']['y'])

    pred_xy_grid = {}
    pred_xy_grid['x'] = np.matrix(data['predictions']['inputs']['x'])
    pred_xy_grid['y'] = np.matrix(data['predictions']['inputs']['y'])

    pred_expected = np.matrix(data['predictions']['expected'])
    pred_mean = np.matrix(data['predictions']['means'])
    pred_sigmas = np.matrix(data['predictions']['sigmas'])

    axis_real = fig.add_subplot(2, 2, 1, projection='3d')
    axis_real.set_title('function to approximate')
    axis_real.scatter(samples_xy_grid['x'], samples_xy_grid['y'], np.matrix(data['train_set']['outputs']), color='green', marker='o', label='train set samples')
    axis_real.plot_surface(pred_xy_grid['x'], pred_xy_grid['y'], pred_expected, color='red', alpha=0.5)
    axis_real.legend()

    axis_pred = fig.add_subplot(2, 2, 2, projection='3d')
    axis_pred.set_title('predicted values using GP')
    axis_pred.plot_surface(pred_xy_grid['x'], pred_xy_grid['y'], pred_mean, color='red', alpha=0.5)

    axis_error = fig.add_subplot(2, 2, 3)
    axis_error.set_title('prediction error')
    shw = axis_error.imshow(pred_mean - pred_expected)
    bar = plt.colorbar(shw)

    axis_sigmas = fig.add_subplot(2, 2, 4)
    axis_sigmas.set_title('standard deviation of the predictions')
    shw = axis_sigmas.imshow(pred_sigmas)
    bar = plt.colorbar(shw)

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
