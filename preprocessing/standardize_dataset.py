import numpy as np
import sys
import os

from common.files import get_files_in_dir, make_dir_if_not_exists
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

IMAGE_HEIGHT = 50
IMAGE_WIDTH = 100
CHANNELS = 3


if __name__ == '__main__':
	argv_len = len(sys.argv)
	print(argv_len)

	if argv_len < 2 or argv_len > 4:
		print('''
	extract.py
		Standardize all images for a faster performance. It make zero mean and std normalization. 

	Usage:
		python standardize_dataset.py [rois_path] [pattern] [output_path]
		
		roi_path            Path to all images
		pattern             (Optional) Filename pattern to match
		output_path         Path for the standardize frames        

	Example:
		python preprocessing/standardize_dataset.py data/target *.jpg data/standardize 
''')
		exit()

	i_path = None
	o_path = None
	pat = '*.jpg'

	if argv_len == 3:
		i_path = sys.argv[1]
		o_path = sys.argv[2]

	if argv_len == 4:
		i_path = sys.argv[1]
		pat = sys.argv[2]
		o_path = sys.argv[3]

	if i_path is None or o_path is None:
		print('Both input and output are required\n')
		exit()

	rois_path = os.path.realpath(i_path)
	output_path = os.path.realpath(o_path)

	train_files = []
	standardize_files = []

	old_standardize_dir = ""
	for file_path in get_files_in_dir(i_path, pat):
		# save the path from train image

		train_files.append(file_path)

		# save the path to the standardize image
		output_standardize_file_path = file_path.replace(rois_path, output_path)
		# make dir for standadize images
		current_standardize_dir = os.path.dirname(os.path.abspath(output_standardize_file_path))
		if current_standardize_dir != old_standardize_dir:
			make_dir_if_not_exists(current_standardize_dir)
		old_standardize_dir = current_standardize_dir

		standardize_files.append(output_standardize_file_path)

	# put image in an array
	dataset = np.ndarray(shape=(len(train_files), IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
						 dtype=np.float32)

	i = 0
	for _file in train_files:
		print(_file)
		img = load_img(_file)  # this is a PIL image
		# img.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT))
		# Convert to Numpy Array
		x = img_to_array(img)
		dataset[i] = x
		i += 1

	# standardize the data with keras
	datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
	datagen.fit(dataset)
	dataset = datagen.standardize(dataset)

	i = 0
	for img_array in dataset:
		standardize_img = array_to_img(img_array)
		standardize_path = standardize_files[i]

		standardize_img.save(standardize_path)

		i += 1
