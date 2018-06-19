import os
import shutil
import sys
sys.path.append("..")
from common.files import make_dir, walklevel
import argparse

def generate_train_val(mouths_path, destination_path, all_aligments_path, train_size, val_size):
	"""
	provider_frames_aligns.py
		Separates the folders where the mouths standardize are into val and train

		Copy the mouths folders to the destination path and separates it into train a val folders depending on the train size and val size

	Usage:
		>>> python provider_frames_aligns.py -i [input_path] -d [destination_path] -ts [train_size] -vs [validation_size] -a [all_aligments_path]

	Example:
		>>> python provider_frames_aligns.py -i ..\data\standardize -d ..\data -ts 10 -vs 6 -a ..\..\dataset\align
	
	:param mouths_path:
	:param destination_path:
	:param train_size: for each person it selects this number of folder to be copied into the train folder
	:param val_size: for each person it selects this number of folder to be copied into the val folder
	"""
	trainig_path = os.path.join(destination_path, "train")
	val_path = os.path.join(destination_path, "val")
	align_path = os.path.join(destination_path, "align")

	make_dir(trainig_path)
	make_dir(val_path)
	make_dir(align_path)

	for subdir, _, _ in walklevel(mouths_path):
		train_count = 0
		val_count = 0

		print(subdir)

		if subdir == mouths_path:
			continue

		for subdir_2, _, _ in walklevel(subdir):

			if subdir_2 == subdir:
				continue
			folder_name = subdir_2.split('\\')[-1]
			origin_align_file = os.path.join(all_aligments_path, folder_name) + ".align"
			dest_align_file = os.path.join(align_path, folder_name) + ".align"

			if train_count < train_size:
				print("Train" + subdir_2)
				train_folder_path = os.path.join(trainig_path, folder_name)

				if not os.path.exists(train_folder_path):
					shutil.copytree(subdir_2, train_folder_path)

				train_count += 1

			elif val_count < val_size:
				print("Val" + subdir_2)

				val_folder_path = os.path.join(val_path, folder_name)

				if not os.path.exists(val_folder_path):
					shutil.copytree(subdir_2, val_folder_path)

				val_count += 1

			if not os.path.isfile(dest_align_file):
				shutil.copyfile(origin_align_file, dest_align_file)

if __name__ == '__main__':
    # TODO: change instead of specify the train number an val number only specify the porcentage of training and have the option to clear the current folders
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input-path", required=True, help="Path to standardize mouths")
	ap.add_argument("-a", "--all-aligments-path", required=True, help="Path to all aligments")
	ap.add_argument("-d", "--destination-path", required=True, help="Path to data")
	ap.add_argument("-ts", "--train-size", required=True, help="Number of train videos", type=int)
	ap.add_argument("-vs", "--val-size", required=True, help="Number of val videos", type=int)

	args = vars(ap.parse_args())

	i_path = args["input_path"]
	all_aligments_path = args["all_aligments_path"]
	d_path = args["destination_path"]
	train_size = args["train_size"]
	val_size = args["val_size"]

	generate_train_val(i_path, d_path, all_aligments_path, train_size, val_size)
