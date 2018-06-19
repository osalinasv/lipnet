import os
import multiprocessing
import numpy as np

from keras.callbacks import Callback
from lipnext.helpers.threadsafe import threadsafe_generator, get_list_safe


class BatchGenerator(Callback):

	def __init__(self, *, dataset_path: str, minibatch_size: int, frame_count: int, image_channels: int, image_height: int, image_width: int, max_string: int):
		self.dataset_path = os.path.realpath(dataset_path)

		self.train_path = os.path.join(self.dataset_path, 'train')
		self.val_path = os.path.join(self.dataset_path, 'val')
		self.align_path = os.path.join(self.dataset_path, 'align')

		self.minibatch_size = minibatch_size
		self.frame_count = frame_count
		self.image_channels = image_channels
		self.image_height = image_height
		self.image_width = image_width
		self.max_string = max_string

		self.cur_train_index = multiprocessing.Value('i', 0)
		self.cur_val_index = multiprocessing.Value('i', 0)

		self.shared_train_epoch = multiprocessing.Value('i', -1)
		self.process_train_index = -1
		self.process_val_index = -1

		self.steps_per_epoch = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
		self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps

		self.val_list = []
		self.train_list = []

		self.build_dataset()

	@threadsafe_generator
	def train_generator(self):
		while True:
			with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
				train_index = self.cur_train_index.value
				self.cur_train_index.value += self.minibatch_size

				if train_index >= self.steps_per_epoch * self.minibatch_size:
					train_index = 0
					self.shared_train_epoch.value += 1
					self.cur_train_index.value = self.minibatch_size

				if self.shared_train_epoch.value < 0:
					self.shared_train_epoch.value += 1

				# Shared index overflow
				if self.cur_train_index.value >= self.training_size:
					self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size

			batch = self.get_batch(train_index, self.minibatch_size,)
			yield batch

	@threadsafe_generator
	def val_generator(self):
		while True:
			with self.cur_val_index.get_lock():
				val_index = self.cur_val_index.value
				self.cur_val_index.value += self.minibatch_size

				if self.cur_val_index.value >= self.validation_size:
					self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size

			batch = self.get_batch(val_index, self.minibatch_size, training=False)
			yield batch

	def on_train_begin(self, logs={}):
		with self.cur_train_index.get_lock():
			self.cur_train_index.value = 0

		with self.cur_val_index.get_lock():
			self.cur_val_index.value = 0

	@property
	def training_size(self) -> int:
		return len(self.train_list)

	@property
	def default_training_steps(self) -> int:
		return self.training_size / self.minibatch_size

	@property
	def validation_size(self) -> int:
		return len(self.val_list)

	@property
	def default_validation_steps(self) -> int:
		return self.validation_size / self.minibatch_size

	def get_output_size(self):
		return 28

	def get_cache_path(self):
		return self.dataset_path.rstrip('/') + '.cache'

	def build_dataset(self):
		pass

	def get_batch(self, index: int, size: int, training: bool = True) -> (dict, dict):
		video_list = self.train_list if training else self.val_list

		X_data_safe = get_list_safe(video_list, index, size)
		X_data = []
		Y_data = []
		label_length = []
		input_length = []

		for path in X_data_safe:
			video_frames = self.get_frames(path)
			align = self.get_align(path)

			X_data.append(video_frames)
			Y_data.append(align)

			# label_length.append( lenght of align sentence )
			# input_length.append( number of frames )

		X_data = np.array(X_data).astype(np.float32) / 255
		Y_data = np.array(Y_data)
		label_length = np.array(label_length)
		input_length = np.array(input_length)

		inputs = {
			'the_input': X_data,
			'the_labels': Y_data,
			'input_length': input_length,
			'label_length': label_length,
		}

		outputs = {'ctc': np.zeros([size])}

		return inputs, outputs

	def get_frames(self, path: str) -> list:
		pass

	def get_align(self, path: str) -> str:
		pass


if __name__ == '__main__':
	generator = BatchGenerator(
		dataset_path='path',
		minibatch_size=50,
		frame_count=30,
		image_channels=3,
		image_height=50,
		image_width=100,
		max_string=32
	)

	for idx, res in enumerate(generator.train_generator()):
		if idx >= 10:
			break

		print(res)
