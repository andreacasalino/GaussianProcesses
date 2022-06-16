import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def import_json(file_name):
    f = open(file_name)
    result = json.load(f)
    return result

def plot_predictions(axis, title, data):
    axis.set_title(title)
    pred_in = data['inputs']
    pred_mean = np.array(data['means'])
    pred_sigmas = np.array(data['sigmas'])
    axis.plot(pred_in, pred_mean, color='red', label='GP predictions mean')
    axis.fill_between(pred_in, 
            pred_mean - 2.5 * pred_sigmas, 
            pred_mean + 2.5 * pred_sigmas, 
        alpha=0.2, color='red', label="GP predictions uncertainty")
    axis.legend()

file_name = 'Log.json'
if(1 < len(sys.argv)):
    file_name = sys.argv[1]

title_passed = ''
if(2 < len(sys.argv)):
    title_passed = sys.argv[2]

data = import_json(file_name)

fig, axis = plt.subplots(1, 2)

plot_predictions(axis[0], 'un-trained model', data['un-trained'])
plot_predictions(axis[1], 'trained model', data['trained'])

plt.show()
