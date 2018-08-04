import numpy as np
import os

from matplotlib import pyplot as plt
from lipnext.helpers.video import get_video_data_from_file


# retrieving

dataset_path = os.path.realpath('./data/dataset/')

last_group = ''
per_group = 2
per_group_counter = 0

batch_size = 45
batch = []

for root, _, files in os.walk(dataset_path):
	if root != last_group:
		per_group_counter = 0

	for f in files:
		if per_group_counter >= per_group:
			break

		if len(batch) >= batch_size:
			break

		if os.path.splitext(f)[1] == '.npy':
			per_group_counter += 1
			batch.append(get_video_data_from_file(os.path.join(root, f)))

	last_group = root

batch = np.array(batch)

print('loaded shape: {}'.format(batch.shape))


# 1st Option. Normalization
# (x - x.min()) / (x.max() - x.min()) # values from 0 to 1

# d_min = batch.min(axis=(2, 3), keepdims=True)
# d_max = batch.max(axis=(2, 3), keepdims=True)
# batch = (batch - d_min) / (d_max - d_min)

# 3rd Option. Standard deviation normalization
# (x - x.mean()) / x.std() # values from ? to ?, but mean at 0

batch = (batch - np.mean(batch)) / np.std(batch)

# displaying

print('normalized shape: {}'.format(batch.shape))

batch = np.swapaxes(batch, 3, 2)

print('swap for viz shape: {}'.format(batch.shape))

img = None

for v in batch:
	for f in v:
		if img is None:
				img = plt.imshow(f)
		else:
				img.set_data(f)

		plt.pause(1 / 25)

plt.show()
