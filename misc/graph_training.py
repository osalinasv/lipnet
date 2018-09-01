import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys

from colorama import init, Fore
from common.files import is_file, get_file_extension


init(autoreset=True)


# set PYTHONPATH=%PYTHONPATH%;./
# python misc\graph_training.py -i data\logs\2018-08-28-00-04-11\2018-08-28-00-04-11_train.csv

# These are the "Tableau 20" colors as RGB.
tableau20 = [
	(31, 119, 180),  (174, 199, 232), (255, 127, 14),  (255, 187, 120),
	(44, 160, 44),   (152, 223, 138), (214, 39, 40),   (255, 152, 150),
	(148, 103, 189), (197, 176, 213), (140, 86, 75),   (196, 156, 148),
	(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
	(188, 189, 34),  (219, 219, 141), (23, 190, 207),  (158, 218, 229)
]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
norm_rgb   = lambda x: x / 255.0
norm_color = lambda x: tuple(map(norm_rgb, x))

tableau20 = list(map(norm_color, tableau20))


def get_data(path: str) -> (np.ndarray, np.ndarray):
	print('CSV file: {}'.format(path))

	with open(path, mode='r') as f:
		reader = csv.DictReader(f, delimiter=',')
		data = [r for r in reader]

	loss = np.array([r['loss'] for r in data], dtype=np.float32)
	val_loss = np.array([r['val_loss'] for r in data], dtype=np.float32)

	return loss, val_loss


def style_axis(ax: plt.Axes, x_max: int, y_max: float):
	# Remove the plot frame lines. They are unnecessary chartjunk.
	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)

	# Ensure that the axis ticks only show up on the bottom and left of the plot.
	# Ticks on the right and top of the plot are generally unnecessary chartjunk.
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	ax.tick_params(axis='both', which='major', labelsize=10)

	# Limit the range of the plot to only where the data is.
	# Avoid unnecessary whitespace.
	ax.set_ylim(0, y_max)
	ax.set_xlim(0, x_max)

	ax.yaxis.set_ticks(np.arange(0, y_max + 5.0, 5.0))
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

	ax.xaxis.set_ticks(range(0, x_max + 10, 10))

	# Provide tick lines across the plot to help your viewers trace along
	# the axis ticks. Make sure that the lines are light and small so they
	# don't obscure the primary data lines.
	ax.grid(b=True, axis='y', alpha=0.3)

	ax.set_ylabel('pérdida', fontsize=10, labelpad=5, weight='bold')
	ax.set_xlabel('epoch', fontsize=10, labelpad=5, weight='bold')

	return ax


def annotate_lowest(dataset: np.ndarray, ax: plt.Axes):
	x = np.argmin(dataset)
	y = dataset[x]

	print('Found lowest value at: ({}, {:.4f})'.format(x, y))

	text = 'val_loss: ({}, {:.3f})'.format(x, y)

	bbox_props = dict(boxstyle="round,pad=0.35", fc="w", ec="k", lw=0.72)
	arrow_props = dict(arrowstyle="-|>", connectionstyle="angle,angleA=0,angleB=60")

	kw = dict(xycoords='data',textcoords="offset pixels", arrowprops=arrow_props, bbox=bbox_props, ha="right", va="top")

	ax.annotate(text, xy=(x, y), xytext=(-35, 60), **kw)


def main():
	print('LipNext Training graphicator\n')

	ap = argparse.ArgumentParser()

	ap.add_argument('-i', '--input-path', required=True,
		help='Path to the training log CSV file')

	ap.add_argument('-o', '--output-path', required=False,
		help='Path to where the output will be saved', default=None)

	args = vars(ap.parse_args())

	input_path = os.path.realpath(args['input_path'])

	if not is_file(input_path) or get_file_extension(input_path) != '.csv':
		print(Fore.RED + '\nERROR: Input path is not a CSV file')
		return

	output_arg = args['output_path']

	if output_arg is None:
		current_dir = os.path.dirname(os.path.realpath(__file__))
		csv_name = os.path.splitext(os.path.basename(input_path))[0]

		output_path = '{}_graph.png'.format(os.path.join(current_dir, csv_name))
	else:
		output_path = os.path.realpath(output_arg)

	loss, val_loss = get_data(input_path)

	dataset = np.array([loss, val_loss])
	x_max = max([len(x) for x in dataset])
	y_max = np.max(dataset)

	# You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
	# exception because of the number of lines being plotted on it.    
	# Common sizes: (10, 7.5) and (12, 9)
	plt.figure(figsize=(10, 7.5))

	ax = plt.subplot(111)

	# Plot each line separately with its own color, using the Tableau 20
	# color set in order.
	plt.plot(loss, label='loss', color=tableau20[1])
	plt.plot(val_loss, label='val_loss', color=tableau20[0])

	ax = style_axis(ax, x_max, y_max)

	annotate_lowest(val_loss, ax)
	ax.legend()

	plt.title('Pérdida a lo largo del entrenamiento', fontsize=16, weight='bold')

	plt.text(0, -8,
		"Autor: Omar Adrian Salinas Villanueva\n"
		"Universidad Autónoma de Ciudad Juárez", fontsize=8, alpha=0.6)

	plt.savefig(output_path, bbox_inches="tight")
	print('Saved output to: {}'.format(output_path))

	plt.show()


if __name__ == '__main__':
	main()
