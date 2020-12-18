import numpy as np

filepath = 'dataset.csv'
data = np.loadtxt(filepath, dtype=np.float, delimiter=',', usecols=range(12), encoding='utf-8')
data_tag = np.loadtxt(filepath, dtype=np.int, delimiter=',', usecols=12, encoding='utf-8')
print(data)
