from matplotlib import pyplot as plt
from matplotlib import patheffects as path_effects


def visualize_video_subtitle(video_frames, subtitle):
	fig, ax = plt.subplots()
	fig.show()

	text = plt.text(0.5, 0.1, "", ha='center', va='center', transform=ax.transAxes,
		fontdict={ 'fontsize': 15, 'color': 'yellow', 'fontweight': 500 })
	text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

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
		fig.canvas.draw()
