import os
import pickle

from common.files import is_file, get_files_in_dir, get_file_name
from lipnext.helpers.align import Align
from lipnext.generators.batch_generator import BatchGenerator


class DatasetGenerator(object):

	def __init__(self, dataset_path: str, batch_size: int, max_string: int, use_cache: bool = True):
		self.dataset_path = os.path.realpath(dataset_path)
		self.batch_size   = batch_size
		self.max_string   = max_string
		self.use_cache    = use_cache

		self.train_path = os.path.join(self.dataset_path, 'train')
		self.val_path   = os.path.join(self.dataset_path, 'val')
		self.align_path = os.path.join(self.dataset_path, 'align')

		self.train_generator = None
		self.val_generator   = None

		# self.steps_per_epoch  = None
		# self.validation_steps = None

		self.build_dataset()


	def build_dataset(self):
		cache_path = self.dataset_path.rstrip('/') + '.cache'

		train_videos = []
		train_aligns = {}

		val_videos   = []
		val_aligns   = {}

		if self.use_cache and is_file(cache_path):
			print('\nLoading dataset list from cache...\n')

			with open(cache_path, 'rb') as f:
				train_videos, train_aligns, val_videos, val_aligns = pickle.load(f)
		else:
			print('\nEnumerating dataset list from disk...\n')

			train_videos = self.get_numpy_files_in_dir(self.train_path)
			train_aligns = self.generate_align_hash(train_videos)

			val_videos = self.get_numpy_files_in_dir(self.val_path)
			val_aligns = self.generate_align_hash(val_videos)

			with open(cache_path, 'wb') as f:
				pickle.dump((train_videos, train_aligns, val_videos, val_aligns), f)

		print('Found {} videos and {} aligns for training'.format(len(train_videos), len(train_aligns)))
		print('Found {} videos and {} aligns for validation\n'.format(len(val_videos), len(val_aligns)))

		self.train_generator = BatchGenerator(train_videos, train_aligns, self.batch_size)
		self.val_generator   = BatchGenerator(val_videos, val_aligns, self.batch_size)


	def get_numpy_files_in_dir(self, path: str) -> list:
		return [f for f in get_files_in_dir(path, '*.npy')]


	def generate_align_hash(self, videos: list) -> dict:
		align_hash = {}

		for path in videos:
			video_name = get_file_name(path)
			align_path = os.path.join(self.align_path, video_name) + '.align'

			align_hash[video_name] = Align(self.max_string).from_file(align_path)

		return align_hash


if __name__ == '__main__':
	datagen = DatasetGenerator('./data/ordered', 50, 32)

	print('train_generator len: {}'.format(len(datagen.train_generator)))
	print('val_generator len:   {}'.format(len(datagen.val_generator)))

	print('\ntrain_generator\n')
	for i, (inp, _) in enumerate(datagen.train_generator):
		if i > 9: break

		print('input: {}'.format(inp['input'].shape))

	print('\nval_generator\n')
	for i, (inp, _) in enumerate(datagen.val_generator):
		if i > 9: break

		print('input: {}'.format(inp['input'].shape))