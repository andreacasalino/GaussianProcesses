import json
import matplotlib.pyplot as plt
import numpy as np

def import_json(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

data = import_json('kernels_log.json')

for log in data:
    plt.matshow(log['kernel'])
    plt.title(log['title'])
plt.show()
