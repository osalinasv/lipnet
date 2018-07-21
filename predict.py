import argparse
import os

from common.files import is_file, get_file_extension


def predict(weights_file_path: str, video_file_path: str):
	print('weights_file_path: {}'.format(weights_file_path))
	print('video_file_path: {}'.format(video_file_path))


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument('-v', '--video-path', required=True,
		help='Path to video file to analize')

	ap.add_argument('-w', '--weights-path', required=True,
		help='Path to .hdf5 trained weights file')

	args = vars(ap.parse_args())

	weights = os.path.realpath(args['weights_path'])
	video = os.path.realpath(args['video_path'])

	if not is_file(weights) or get_file_extension(weights) != '.hdf5':
		print('Invalid path to trained weights file')
		return

	if not is_file(video):
		print('Invalid path to video file')
		return
	
	predict(weights, video)


if __name__ == '__main__':
	main()
