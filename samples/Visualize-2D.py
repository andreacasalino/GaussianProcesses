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
    axis = fig.add_subplot(111, projection='3d')

    plt.title(title)

    axis.scatter(np.matrix(data['train_set']['inputs']['x']), np.matrix(data['train_set']['inputs']['y']), np.matrix(data['train_set']['outputs']), color='green', marker='o')

    pred_xy_grid = {}
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
