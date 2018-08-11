import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


plt.style.use('seaborn-whitegrid')

_, ax = plt.subplots()

csv_file_path = os.path.realpath(sys.argv[1])
print('csv file: {}\n'.format(csv_file_path))

with open(csv_file_path, mode='r') as f:
	reader = csv.DictReader(f, delimiter=',')
	data = [r for r in reader]

loss = np.array([r['loss'] for r in data], dtype=np.float32)
val_loss = np.array([r['val_loss'] for r in data], dtype=np.float32)

ax.plot(loss, label='loss')
ax.plot(val_loss, label='val_loss')

ax.legend(loc='upper right')

plt.show()
