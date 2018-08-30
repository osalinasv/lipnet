import numpy as np

from matplotlib import pyplot as plt
from matplotlib import patheffects as path_effects


FRAME_RATE = 1 / 25


def visualize_video_subtitle(video_frames: np.ndarray, subtitle: str, swap_axes: bool = True):
	_, ax = plt.subplots()

	text = plt.text(0.5, 0.1, "", ha='center', va='center', transform=ax.transAxes,
		fontdict={ 'fontsize': 15, 'color': 'yellow', 'fontweight': 500 })
	text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

	video_frames = np.swapaxes(video_frames, 1, 2)

	subs = subtitle.split()
	inc = max(len(video_frames) / (len(subs) + 1), 0.01)

	img = None

	for i, frame in enumerate(video_frames):
		sub = " ".join(subs[:int(i/inc)])
		text.set_text(sub)

		if img is None:
				img = plt.imshow(frame)
		else:
				img.set_data(frame)

		plt.pause(FRAME_RATE)

	plt.show()
