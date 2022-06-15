import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def import_json(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

def plot_mat(mat, title):
    fig, ax = plt.subplots(1)
    fig.suptitle(title)
    shw = ax.imshow(mat)
    bar = plt.colorbar(shw)

def print_log(title, data):
    plot_mat(data['likelihood'], 'likelihood')
    c = 1
    for grad in data['likelihood_gradient']:
        plot_mat(grad, 'gradient at ' + str(c))
        c += 1

data = import_json('gradients_log.json')

if len(sys.argv) == 1:
    for log in data:
        print_log(log, data[log])
else:
    print_log(sys.argv[1], log[sys.argv[1]])
plt.show()
