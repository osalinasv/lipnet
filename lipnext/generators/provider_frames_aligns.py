import os
import shutil
import sys
from common.files import make_dir
import argparse

sys.path.append("..\..")

def generate_train_val(mouths_path, destination_path, train_size, val_size):
	"""
	provider_frames_aligns.py
		Separates the folders where the mouths standardize are into val and train

		Copy the mouths folders to the destination path and separates it into train a val folders depending on the train size and val size

	Usage:
		>>> python provider_frames_aligns.py -i [input_path] -d [destination_path] -ts [train_size] -vs [validation_size]

	Example:
		>>> python provider_frames_aligns.py -i ..\..\data\standardize -d ..\..\data -ts 10 -vs 6
	
	:param mouths_path:
	:param destination_path:
	:param train_size: for each person it selects this number of folder to be copied into the train folder
	:param val_size: for each person it selects this number of folder to be copied into the val folder
	"""
	trainig_path = os.path.join(destination_path, "train")
	val_path = os.path.join(destination_path, "val")

	make_dir(trainig_path)
	make_dir(val_path)

	for subdir, _, _ in walklevel(mouths_path):
		train_count = 0
		val_count = 0

		print(subdir)

		if subdir == mouths_path:
			continue

		for subdir_2, _, _ in walklevel(subdir):
			if subdir_2 == subdir:
				continue

			if train_count < train_size:
				print("Train" + subdir_2)
				train_folder_path = os.path.join(trainig_path, subdir_2.split('\\')[-1])

				if not os.path.exists(train_folder_path):
					shutil.copytree(subdir_2, train_folder_path)

				train_count += 1

			elif val_count < val_size:
				print("Val" + subdir_2)

				val_folder_path = os.path.join(val_path, subdir_2.split('\\')[-1])

				if not os.path.exists(val_folder_path):
					shutil.copytree(subdir_2, val_folder_path)

				val_count += 1

def walklevel(some_dir, level=1):
	some_dir = some_dir.rstrip(os.path.sep)

	assert os.path.isdir(some_dir)

	num_sep = some_dir.count(os.path.sep)

	for root, dirs, files in os.walk(some_dir):
		yield root, dirs, files

		num_sep_this = root.count(os.path.sep)

		if num_sep + level <= num_sep_this:
			del dirs[:]

if __name__ == '__main__':
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input-path", required=True, help="Path to standardize mouths")
	ap.add_argument("-d", "--destination-path", required=True, help="Path to data")
	ap.add_argument("-ts", "--train-size", required=True, help="Number of train videos", type=int)
	ap.add_argument("-vs", "--val-size", required=True, help="Number of val videos", type=int)

	args = vars(ap.parse_args())

	i_path = args["input_path"]
	d_path = args["destination_path"]
	train_size = args["train_size"]
	val_size = args["val_size"]

	generate_train_val(i_path, d_path, train_size, val_size)
