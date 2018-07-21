import os
import sys

from common.files import is_file


def predict(weights_file_path: str, video_file_path: str):
	print('weights_file_path: {}'.format(weights_file_path))
	print('video_file_path: {}'.format(video_file_path))


def main():
	if len(sys.argv) != 2:
		print('\nUsage:\n$ python predict.py <path_to_weights> <path_to_video>')
		return

	weights = os.path.realpath(sys.argv[1])
	video = os.path.realpath(sys.argv[2])

	if is_file(weights) and is_file(video):
		predict(weights, video)


if __name__ == '__main__':
	main()
