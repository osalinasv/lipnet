import argparse
import math
import os
import random
import shutil

from common.files import make_dir_if_not_exists, walklevel, get_file_name, get_files_in_dir


#  python preprocessing\structure_dataset.py -d data\dataset\ -o data\ordered -a D:\GRID\align\
def structure_dataset(dataset_path: str, output_path: str, alignments_path: str, validation_split: float):
	dataset_path    = os.path.realpath(dataset_path)
	output_path     = os.path.realpath(output_path)
	alignments_path = os.path.realpath(alignments_path)

	train_list = []
	val_list   = []
	align_list = []

	for sub_dir, _, _ in walklevel(dataset_path):
		if sub_dir == dataset_path:
			continue

		sub_dir_name = os.path.basename(sub_dir)

		videos_in_group = [v for v in get_files_in_dir(sub_dir, '*.npy')]
		random.shuffle(videos_in_group)

		videos_in_group_len = len(videos_in_group)
		validation_amount = math.floor(videos_in_group_len * validation_split)
		train_amount = videos_in_group_len - validation_amount

		print('Speaker {}: total {} - train {} - val {}'.format(sub_dir_name, videos_in_group_len, train_amount, validation_amount))

		train_list += videos_in_group[:train_amount]
		val_list   += videos_in_group[train_amount:]
		align_list += [get_file_name(a) + '.align' for a in videos_in_group]

	target_train_path = os.path.realpath(os.path.join(output_path, "train"))
	make_dir_if_not_exists(target_train_path)

	for tp in train_list:
		shutil.copyfile(tp, os.path.join(target_train_path, os.path.basename(tp)))

	target_val_path = os.path.realpath(os.path.join(output_path, "val"))
	make_dir_if_not_exists(target_val_path)

	for vp in val_list:
		shutil.copyfile(vp, os.path.join(target_val_path, os.path.basename(vp)))

	target_align_path = os.path.realpath(os.path.join(output_path, "align"))
	make_dir_if_not_exists(target_align_path)

	for ap in align_list:
		shutil.copyfile(os.path.join(alignments_path, ap), os.path.join(target_align_path, ap))

	print('\nStructuring finalized')


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument('-d', '--dataset-path', required=True,
		help='Path to the unstructured/unordered dataset')

	ap.add_argument('-o', '--output-path', required=True,
		help='Path to the structured dataset')

	ap.add_argument('-a', '--alignments-path', required=True,
		help='Path to a directory containing all alignment files')

	ap.add_argument('-vs', '--validation-split', required=False,
		help='(Optional) Percentage of the dataset to reserve for validation', type=float, default=0.2)

	args = vars(ap.parse_args())

	dataset_path     = args['dataset_path']
	output_path      = args['output_path']
	alignments_path  = args['alignments_path']
	validation_split = args['validation_split']

	structure_dataset(dataset_path, output_path, alignments_path, validation_split)


if __name__ == '__main__':
	main()
