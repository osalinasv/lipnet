import numpy as np

from common.files import get_file_name
from lipnext.helpers.video import get_video_data_from_file
from keras.utils import Sequence


class BatchGenerator(Sequence):

	def __init__(self, video_paths: list, align_hash: dict, batch_size: int):
		self.video_paths = video_paths
		self.align_hash = align_hash
		self.batch_size  = batch_size

		self.videos_len = len(self.video_paths)


	def __len__(self) -> int:
		return int(np.ceil(self.videos_len / float(self.batch_size)))


	def __getitem__(self, idx: int) -> (dict, dict):
		split_start = idx * self.batch_size
		split_end   = split_start + self.batch_size

		if split_end > self.videos_len:
			split_end = self.videos_len

		videos_batch = self.video_paths[split_start:split_end]

		x_data = []
		y_data = []
		input_length = []
		label_length = []

		for path in videos_batch:
			video_data, label, label_len = self.get_data_from_path(path)

			x_data.append(video_data)
			y_data.append(label)
			label_length.append(label_len)
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

		real_batch_size = len(x_data)
		outputs = { 'ctc': np.zeros([real_batch_size]) } # dummy data for dummy loss function

		return inputs, outputs


	def get_data_from_path(self, path: str) -> (np.ndarray, np.ndarray, int, int):
		video_data = get_video_data_from_file(path)
		align_data = self.align_hash[get_file_name(path)]

		return video_data, align_data.padded_label, align_data.label_length


	def flip_frame(self, frame: np.ndarray) -> np.ndarray:
		return np.fliplr(frame)


	def flip_video(self, video_data: np.ndarray) -> np.ndarray:
		return np.array([self.flip_frame(f) for f in video_data])
