import numpy as np
from progress.bar import ShadyBar

import env
from common.files import get_files_in_dir
from core.helpers.video import get_video_data_from_file


# mean [0.68873879 0.50568776 0.3343385]
# std  [0.14589616 0.11136819 0.11655929]


PATH = './data/dataset'
SIZE = 28620

print('\nFINDING MEAN\n')

bar = ShadyBar('videos', max=SIZE)

# T x W x H x C
videos_sum   = np.zeros((env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, env.IMAGE_CHANNELS))
videos_count = 0

for video_path in get_files_in_dir(PATH, '*.npy'):
	videos_sum   += get_video_data_from_file(video_path)
	videos_count += 1
	bar.next()

bar.finish()

mean_divide = np.vectorize(lambda x: x / videos_count)
videos_sum  = mean_divide(videos_sum)

print('\nshape of sum: {}\n'.format(videos_sum.shape))

rgb_mean = np.mean(videos_sum, axis=(0, 1, 2))
print('rgb mean: {}    {}'.format(rgb_mean.shape, rgb_mean))

mean_video = np.tile(rgb_mean, (env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, 1))
print('mean video shape: {}\n'.format(mean_video.shape))

print('\nFINDING STD\n')

bar = ShadyBar('videos', max=SIZE)

# T x W x H x C
videos_sum   = np.zeros((env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, env.IMAGE_CHANNELS))

for video_path in get_files_in_dir(PATH, '*.npy'):
	videos_sum += np.power(np.absolute(get_video_data_from_file(video_path) - mean_video), 2)
	bar.next()

bar.finish()

videos_sum = mean_divide(videos_sum)
videos_sum = np.sqrt(videos_sum)

print('\nshape of sum: {}\n'.format(videos_sum.shape))

rgb_std = np.mean(videos_sum, axis=(0, 1, 2))
print('rgb std: {}    {}'.format(rgb_std.shape, rgb_std))
