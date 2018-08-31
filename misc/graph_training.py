import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
						 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
						 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
						 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
						 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
norm_rgb   = lambda x: x / 255.0
norm_color = lambda x: tuple(map(norm_rgb, x))

tableau20 = list(map(norm_color, tableau20))

# You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
# exception because of the number of lines being plotted on it.    
# Common sizes: (10, 7.5) and (12, 9)
plt.figure(figsize=(10, 7.5))

ax = plt.subplot(111)

# Remove the plot frame lines. They are unnecessary chartjunk.
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Obtain the data from the CSV
csv_file_path = os.path.realpath(sys.argv[1])
print('csv file: {}\n'.format(csv_file_path))

with open(csv_file_path, mode='r') as f:
	reader = csv.DictReader(f, delimiter=',')
	data = [r for r in reader]

loss = np.array([r['loss'] for r in data], dtype=np.float32)
val_loss = np.array([r['val_loss'] for r in data], dtype=np.float32)

loss_range = np.min(loss), np.max(loss)
val_loss_range = np.min(val_loss), np.max(val_loss)

y_range = min(loss_range[0], val_loss_range[0]), max(loss_range[1], val_loss_range[1])
x_range = 0, max(len(val_loss), len(loss))

# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
plt.ylim(y_range[0], y_range[1])
plt.xlim(x_range[0], x_range[1]) 

plt.yticks(range(int(y_range[0]), int(y_range[1]) + 10, 10), fontsize=10)
plt.xticks(range(int(x_range[0]), int(x_range[1]) + 10, 10), fontsize=10)

# Provide tick lines across the plot to help your viewers trace along
# the axis ticks. Make sure that the lines are light and small so they
# don't obscure the primary data lines.
helper_line_range_x = range(x_range[0], x_range[1])

for y in range(int(y_range[0]), int(y_range[1]) + 10, 10):
	plt.plot(helper_line_range_x, [y] * len(helper_line_range_x), "--", lw=0.5, color="black", alpha=0.3)

# Remove the tick marks; they are unnecessary with the tick lines we just plotted.
plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Plot each line separately with its own color, using the Tableau 20
# color set in order.
plt.plot(loss, label='loss', lw=2.5, color=tableau20[14])
plt.plot(val_loss, label='val_loss', lw=2.5, color=tableau20[0])

# Again, make sure that all labels are large enough to be easily read
# by the viewer.
plt.text(x_range[1] - 2, y_range[1] - 2, 'Loss', fontsize=14, color=tableau20[14], ha='right')
plt.text(x_range[1] - 2, y_range[1] - 5, 'Validation loss', fontsize=14, color=tableau20[0], ha='right')

plt.text(np.mean(x_range), y_range[1] + 3, "Training loss over time (Epochs)", fontsize=16, ha="center")

plt.text(x_range[0], y_range[0] - 8,
	"Author: Omar Adrian Salinas Villanueva"
	"\nNote: Data values captured using Keras' CSVLogger and ModelCheckpoint", fontsize=8, alpha=0.6)   

csv_name = os.path.splitext(os.path.basename(csv_file_path))[0]
plt.savefig('{}_graph.png'.format(csv_name), bbox_inches="tight")

plt.show()
