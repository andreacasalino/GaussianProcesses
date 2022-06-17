import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def import_json(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

def print_log(log):
    fig, ax = plt.subplots()
    plt.title(log['title'])
    shw = ax.imshow(log['kernel'])
    bar = plt.colorbar(shw)

def print_logs(logs):
    for log in logs:
        print_log(log)

data = import_json('kernels_log.json')
if 1 < len(sys.argv):
    print_logs(data[sys.argv[1]])
else:
    for tag in data:
        print_logs(data[tag])
plt.show()
