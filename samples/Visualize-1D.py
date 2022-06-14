import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def import_json(file_name):
    f = open(file_name)
    result = json.load(f)
    return result

def plot_result(title, data):
    fig, axis = plt.subplots(2, 2)
    fig.suptitle(title)

    pred_in = np.array(data['predictions']['inputs'])

    axis[0][0].set_title('function to approximate')
    axis[0][0].scatter(np.array(data['train_set']['inputs']), np.array(data['train_set']['outputs']), color='green', marker='o', label='train set')
    axis[0][0].plot(pred_in, np.array(data['predictions']['expected']), color='red')
    axis[0][0].legend()

    axis[0][1].set_title('Gaussian process predictions')
    pred_mean = np.array(data['predictions']['means'])
    pred_sigmas = np.array(data['predictions']['sigmas'])
    axis[0][1].scatter(np.array(data['train_set']['inputs']), np.array(data['train_set']['outputs']), color='green', marker='o', label='train set')
    axis[0][1].plot(pred_in, pred_mean, color='red', label='GP predictions mean')
    axis[0][1].fill_between(pred_in, 
            pred_mean - 2.5 * pred_sigmas, 
            pred_mean + 2.5 * pred_sigmas, 
        alpha=0.2, color='red', label="GP predictions uncertainty")
    axis[0][1].legend()

    axis[1][0].set_title('Prediction error')
    axis[1][0].plot(pred_in, pred_mean - np.array(data['predictions']['expected']))

    axis[1][1].set_title('Prediction uncertainty')
    axis[1][1].plot(pred_in, np.array(data['predictions']['sigmas']))


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
