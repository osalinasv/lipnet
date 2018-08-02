import skvideo.io
import numpy as np

from matplotlib import pyplot as plt


def flip_frame(frame: np.ndarray) -> np.ndarray:
	return np.fliplr(frame)


video_data = skvideo.io.vread('D:/GRID/s1/bbaf2n.mpg')
f_video_data = np.array([flip_frame(f) for f in video_data])

print(video_data.shape)
print(f_video_data.shape)

FRAME_RATE = 1 / 25

_, ax = plt.subplots()
img = None

for f in f_video_data:
	if img is None:
				img = plt.imshow(f)
	else:
			img.set_data(f)

	plt.pause(FRAME_RATE)

plt.show()
