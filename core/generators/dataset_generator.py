import os
import pickle
import random

from common.files import get_file_name, get_files_in_dir, get_immediate_subdirs, is_file
from core.generators.batch_generator import BatchGenerator
from core.helpers.align import Align, align_from_file


class DatasetGenerator(object):

	def __init__(self, dataset_path: str, aligns_path: str, batch_size: int, max_string: int, val_split: float, use_cache: bool = True):
		self.dataset_path = os.path.realpath(dataset_path)
		self.aligns_path  = os.path.realpath(aligns_path)
		self.batch_size   = batch_size
		self.max_string   = max_string
		self.val_split    = val_split
		self.use_cache    = use_cache

		self.train_generator = None
		self.val_generator   = None

		self.build_dataset()


	def build_dataset(self):
		cache_path = self.dataset_path.rstrip('/') + '.cache'

		if self.use_cache and is_file(cache_path):
			print('\nLoading dataset list from cache...\n')

			with open(cache_path, 'rb') as f:
				train_videos, train_aligns, val_videos, val_aligns = pickle.load(f)
		else:
			print('\nEnumerating dataset list from disk...\n')

			groups = self.get_speaker_groups(self.dataset_path)
			train_videos, val_videos = self.split_speaker_groups(groups, self.val_split)

			train_aligns = self.generate_align_hash(train_videos)
			val_aligns   = self.generate_align_hash(val_videos)

			with open(cache_path, 'wb') as f:
				pickle.dump((train_videos, train_aligns, val_videos, val_aligns), f)

		print('Found {} videos and {} aligns for training'.format(len(train_videos), len(train_aligns)))
		print('Found {} videos and {} aligns for validation\n'.format(len(val_videos), len(val_aligns)))

		self.train_generator = BatchGenerator(train_videos, train_aligns, self.batch_size)
		self.val_generator   = BatchGenerator(val_videos, val_aligns, self.batch_size)


	@staticmethod
	def get_numpy_files_in_dir(path: str) -> list:
		return [f for f in get_files_in_dir(path, '*.npy')]


	def get_speaker_groups(self, path: str) -> list:
		speaker_groups = []

		for sub_dir in get_immediate_subdirs(path):
			videos_in_group = self.get_numpy_files_in_dir(sub_dir)
			random.shuffle(videos_in_group)

			speaker_groups.append(videos_in_group)

		return speaker_groups


	@staticmethod
	def split_speaker_groups(groups: list, val_split: float) -> (list, list):
		train_list = []
		val_list   = []

		for group in groups:
			group_len  = len(group)
			val_amount = int(round(group_len * val_split))

			train_list += group[val_amount:]
			val_list   += group[:val_amount]

		return train_list, val_list


	def generate_align_hash(self, videos: list) -> {str: Align}:
		align_hash = {}

		for path in videos:
			video_name = get_file_name(path)
			align_path = os.path.join(self.aligns_path, video_name) + '.align'

			align_hash[video_name] = align_from_file(align_path, self.max_string)

		return align_hash
