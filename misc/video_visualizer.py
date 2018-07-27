import numpy as np
import os
import sys

from matplotlib import pyplot as plt


video_data = np.load(os.path.realpath(sys.argv[1]))

for f in video_data:
	plt.imshow(f)
	plt.show()
