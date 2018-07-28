import numpy as np

from common.files import get_file_name
from lipnext.helpers.video import get_video_data_from_file
from keras.utils import Sequence


class BatchGenerator(Sequence):

	def __init__(self, video_paths: list, align_hash: dict, batch_size: int):
		self.video_paths = video_paths
		self.align_hash = align_hash
		self.batch_size  = batch_size


	def __len__(self) -> int:
		return int(np.ceil(len(self.video_paths) / float(self.batch_size)))


	def __getitem__(self, idx: int) -> (dict, dict):
		videos_batch = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

		x_data = []
		y_data = []
		input_length = []
		label_length = []

		for path in videos_batch:
			video_data = get_video_data_from_file(path)
			align_data = self.align_hash[get_file_name(path)]

			x_data.append(video_data)
			y_data.append(align_data.padded_label)
			label_length.append(align_data.label_length)
			input_length.append(len(video_data))

		x_data = np.array(x_data)
		y_data = np.array(y_data)
		input_length = np.array(input_length)
		label_length = np.array(label_length)

		inputs = {
			'input'       : x_data,
			'labels'      : y_data,
			'input_length': input_length,
			'label_length': label_length,
		}

		outputs = { 'ctc': np.zeros([self.batch_size]) } # dummy data for dummy loss function

		return inputs, outputs
