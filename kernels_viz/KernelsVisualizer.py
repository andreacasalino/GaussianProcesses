import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def import_json(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

def show_log(tag):
    if(len(sys.argv) == 1):
        return True
    for arg in sys.argv:
        if(arg == tag):
            return True
    return False

print('arguments:', sys.argv)

data = import_json('kernels_log.json')

for log in data:
    if not show_log(log['tag']):
        continue;
    plt.title(log['title'])
    fig, ax = plt.subplots()
    shw = ax.imshow(log['kernel'])
    bar = plt.colorbar(shw)
plt.show()
