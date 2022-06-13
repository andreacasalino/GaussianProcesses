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

    axis.scatter(np.array(data['train_set']['inputs']), np.array(data['train_set']['outputs']), color='red', marker='o', label='train set')
    pred_in = np.array(data['predictions']['inputs'])
    pred_mean = np.array(data['predictions']['means'])
    pred_sigmas = np.array(data['predictions']['sigmas'])
    axis.plot(pred_in, pred_mean, color='red', label='GP predictions')
    axis.fill_between(pred_in, 
            pred_mean - 2.5 * pred_sigmas, 
            pred_mean + 2.5 * pred_sigmas, 
        alpha=0.2, color='red', label="GP prediction uncertainty")

    axis.legend()

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
