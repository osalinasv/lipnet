import multiprocessing
import numpy as np
import os
import pickle
import sys

from common.files import is_file, get_file_name, get_files_in_dir
from lipnext.helpers.threadsafe import threadsafe_generator
from lipnext.helpers.align import Align
from keras import backend as K
from keras.callbacks import Callback
from scipy import ndimage


class BatchGenerator(Callback):

	def __init__(self, dataset_path: str, minibatch_size: int, frame_count: int, image_channels: int, image_height: int, image_width: int, max_string: int, steps_per_epoch: int = None, validation_steps: int = None):
		self.data_path = os.path.realpath(dataset_path)

		self.train_path = os.path.join(self.data_path, 'train')
		self.val_path   = os.path.join(self.data_path, 'val')
		self.align_path = os.path.join(self.data_path, 'align')

		self.minibatch_size = minibatch_size
		self.frame_count    = frame_count
		self.image_channels = image_channels
		self.image_height   = image_height
		self.image_width    = image_width
		self.max_string     = max_string

		self.cur_train_index = multiprocessing.Value('i', 0)
		self.cur_val_index   = multiprocessing.Value('i', 0)

		self.shared_train_epoch  = multiprocessing.Value('i', -1)
		self.process_train_index = -1
		self.process_val_index   = -1

		self.val_list   = []
		self.train_list = []
		self.align_hash = {}
		
		self.steps_per_epoch  = self.default_training_steps if steps_per_epoch is None else steps_per_epoch
		self.validation_steps = self.default_validation_steps if validation_steps is None else validation_steps

		self.build_dataset()


	def build_dataset(self):
		cache_path = self.cache_path

		if is_file(cache_path):
			print('\nLoading dataset list from cache...\n')

			with open(cache_path, 'rb') as f:
				self.train_list, self.val_list, self.align_hash = pickle.load(f)

		else:
			print('\nEnumerating dataset list from disk...\n')

			self.train_list = self.get_numpy_files_in_dir(self.train_path)
			self.val_list   = self.get_numpy_files_in_dir(self.val_path)
			self.align_hash = self.generate_align_hash(self.train_list + self.val_list)

			with open(cache_path, 'wb') as f:
				pickle.dump((self.train_list, self.val_list, self.align_hash), f)


	@staticmethod
	def get_numpy_files_in_dir(path: str) -> list:
		return [f for f in get_files_in_dir(path, '*.npy')]


	def generate_align_hash(self, video_list: list) -> dict:
		align_hash = {}

		for video_path in video_list:
			video_name = get_file_name(video_path)
			align_path = os.path.join(self.align_path, video_name) + '.align'

			align_hash[video_name] = Align(self.max_string).from_file(align_path)

		return align_hash


	def read_dataset(self, index: int, size: int, train: bool) -> (dict, dict):
		video_list  = self.train_list if train else self.val_list
		X_data_path = self.get_sublist(video_list, index, size)

		X_data = []
		Y_data = []
		label_length = []
		input_length = []

		for path in X_data_path:
			video_data = self.get_video_data_from_file(path)
			align      = self.align_hash[get_file_name(path)]

			X_data.append(video_data)
			Y_data.append(align.padded_label)
			label_length.append(align.label_length)
			input_length.append(len(video_data))

		X_data = np.array(X_data) # TODO: @Error this returns a np array with shape (50,) instead of the needed 5D input shape
		Y_data = np.array(Y_data)
		label_length = np.array(label_length)
		input_length = np.array(input_length)

		inputs = {
			'input'       : X_data,
			'labels'      : Y_data,
			'input_length': input_length,
			'label_length': label_length,
		}

		outputs = { 'ctc': np.zeros([size]) } # dummy data for dummy loss function

		return (inputs, outputs)


	@staticmethod
	def get_sublist(l: list, index: int, size: int) -> list:
		ret = l[index:index + size]

		while size - len(ret) > 0:
			ret += l[0:size - len(ret)]

		return ret


	def get_video_data_from_file(self, path: str):
		video_data = np.load(path).astype(np.float32) / 255 # T x H x W x C
		reshaped_video_data = [self.reshape_video_frame(frame) for frame in video_data] # T x W x H x C

		if K.image_data_format() == 'channels_first':
			reshaped_video_data = np.rollaxis(reshaped_video_data, 3) # C x T x W x H

		return np.array(reshaped_video_data)


	@staticmethod
	def reshape_video_frame(frame):
		frame = frame.swapaxes(0, 1) # swap width and height to form format W x H x C

		if len(frame.shape) < 3:
			frame = np.array([frame]).swapaxes(0, 2).swapaxes(0, 1)  # Add grayscale channel

		return np.array(frame)


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

			yield self.read_dataset(train_index, self.minibatch_size, train=True)


	@threadsafe_generator
	def val_generator(self):
		while True:
			with self.cur_val_index.get_lock():
				val_index = self.cur_val_index.value
				self.cur_val_index.value += self.minibatch_size

				if self.cur_val_index.value >= self.validation_size:
					self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size

			yield self.read_dataset(val_index, self.minibatch_size, train=False)


	def on_train_begin(self, logs={ }):
		with self.cur_train_index.get_lock():
			self.cur_train_index.value = 0

		with self.cur_val_index.get_lock():
			self.cur_val_index.value = 0


	@property
	def cache_path(self) -> str:
		return self.data_path.rstrip('/') + '.cache'


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


if __name__ == '__main__':
	generator = BatchGenerator(
		dataset_path   = './data/ordered',
		minibatch_size = 50,
		frame_count    = 75,
		image_channels = 3,
		image_height   = 50,
		image_width    = 100,
		max_string     = 32
	)

	for idx, res in enumerate(generator.train_generator()):
		if idx > 2: break

		i, _ = res

		print('input:        {}'.format(i['input'].shape))
		print('input inner:  {}'.format(i['input'][0].shape))
		print('labels:       {}'.format(i['labels'].shape))
		print('input_length: {}'.format(i['input_length']))
		print('label_length: {}'.format(i['label_length']))
