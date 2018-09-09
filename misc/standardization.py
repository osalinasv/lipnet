import env
import matplotlib.pyplot as plt
import numpy as np
import time

from core.helpers.video import get_video_data_from_file


# set PYTHONPATH=%PYTHONPATH%;./

# With tiling

# reshaping: 0.49866604804992676
# standardizing: 0.6103672981262207
# sample: [1.55181094 2.3244572  2.44974049]

# No tiling

# standardizing: 0.703120231628418
# sample: [1.55181094 2.3244572  2.44974049]

mean = np.array([env.MEAN_R, env.MEAN_G, env.MEAN_B])
std  = np.array([env.STD_R, env.STD_G, env.STD_B])

print('mean: {}\ts: {}'.format(mean, mean.shape))
print('std:  {}\ts: {}'.format(std, std.shape))
print()

# start_time = time.time()

# video_shape = (env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, 1)

# mean_vid = np.tile(mean, video_shape)
# std_vid  = np.tile(std, video_shape)

# print('mean_vid: {}'.format(mean_vid.shape))
# print('std_vid:  {}'.format(std_vid.shape))
# print()

# mean_batch = np.tile(mean_vid, (env.BATCH_SIZE, 1, 1, 1, 1))
# std_batch  = np.tile(std_vid, (env.BATCH_SIZE, 1, 1, 1, 1))

# print('mean_batch: {}'.format(mean_batch.shape))
# print('std_batch:  {}'.format(std_batch.shape))
# print()

# end_time = time.time()

# print('reshaping: {}'.format(end_time - start_time))

video_data   = get_video_data_from_file('data/dataset_removed/s1/bbaf2n.npy')
video_batch = np.tile(video_data, (env.BATCH_SIZE, 1, 1, 1, 1))

start_time = time.time()

standardized = (video_batch - mean) / std

end_time = time.time()

print('standardizing: {}'.format(end_time - start_time))
print('sample: {}'.format(standardized[0][1][2][3]))

video_data   = np.swapaxes(video_batch[20], 1, 2)
standardized = np.swapaxes(standardized[20], 1, 2)

print('video_data:   {}'.format(video_data.shape))
print('standardized: {}'.format(standardized.shape))
print()

p1 = plt.subplot(211)
p2 = plt.subplot(212)

img1 = None
img2 = None

FRAME_RATE = 1 / 25

for i in range(len(video_data)):
	if img1 is None:
		img1 = p1.imshow(video_data[i], interpolation='none')
	else:
		img1.set_data(video_data[i])

	if img2 is None:
		img2 = p2.imshow(standardized[i], interpolation='none')
	else:
		img2.set_data(standardized[i])

	plt.pause(FRAME_RATE)

plt.show()
