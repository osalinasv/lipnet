import argparse
import dlib
import numpy as np
import os

from colorama import init, Back, Fore, Style
from common.files import is_dir, is_file, get_files_in_dir, make_dir_if_not_exists
from preprocessing.extractor.extract_roi import video_to_frames


init(autoreset=True)


# python preprocessing\extract.py -v D:\GRID\ -o data\dataset -p bbaf*.mpg
def extract_to_npy(videos_path: str, output_path: str, predictor_path: str, pattern: str, first_video: int, last_video: int):
	videos_path = os.path.realpath(videos_path)
	output_path = os.path.realpath(output_path)
	predictor_path = os.path.realpath(predictor_path)

	print('\nEXTRACT\n')
	print('Searching for files in: {}\nMatch for: {}'.format(videos_path, pattern))

	videos_failed = []

	VIDEOS_FAILED_PATH  = os.path.join(output_path, 'videos_failed.log')

	if is_file(VIDEOS_FAILED_PATH):
		with open(VIDEOS_FAILED_PATH, 'r+') as f:
			for line in f.readlines():
				videos_failed.append(line.rstrip())

	last_group_dir_name = ''
	video_count_per_group = 0

	detector  = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	for file_path in get_files_in_dir(videos_path, pattern):
		group_dir_name = os.path.basename(os.path.dirname(file_path))
		video_file_name = os.path.splitext(os.path.basename(file_path))[0]

		video_target_dir  = os.path.join(output_path, group_dir_name)
		video_target_path = os.path.join(video_target_dir, video_file_name) + '.npy'

		if video_target_path in videos_failed:
			print(Fore.RED + 'Video {} is probably corrupted and was ignored'.format(video_file_name))
			continue

		if group_dir_name == last_group_dir_name:
			video_count_per_group += 1
		else:
			video_count_per_group = 0

		last_group_dir_name = group_dir_name

		if video_count_per_group < first_video or video_count_per_group >= last_video:
			continue

		if is_file(video_target_path):
			print(Style.DIM + Fore.CYAN + 'Video {} is already at: {}'.format(video_file_name, video_target_path))
			continue

		make_dir_if_not_exists(video_target_dir)

		if not video_to_frames(file_path, video_target_path, detector, predictor):
			if video_count_per_group > 0:
				video_count_per_group -= 1
			else:
				video_count_per_group = 0

			with open(VIDEOS_FAILED_PATH, 'a+') as f:
				f.write(video_target_path + '\n')

	print('\nExtraction finished')


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument("-v",  "--videos-path", required=True,
		help="Path to videos directory")

	ap.add_argument("-o",  "--output-path", required=True, 
		help="Path for the extracted frames")

	DEFAULT_PREDICTOR = os.path.join(__file__, '..', '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat')

	ap.add_argument("-pp", "--predictor-path", required=False,
		help="(Optional) Path to the predictor .dat file", default=DEFAULT_PREDICTOR)

	ap.add_argument("-p",  "--pattern", required=False,
		help="(Optional) File name pattern to match", default='*.mpg')

	ap.add_argument("-fv", "--first-video", required=False,
		help="(Optional) First video index extracted in each speaker (inclusive)", type=int, default=0)

	ap.add_argument("-lv", "--last-video",  required=False,
		help="(Optional) Last video index extracted in each speaker (exclusive)", type=int, default=1001)

	args = vars(ap.parse_args())

	videos_path    = args["videos_path"]
	output_path    = args["output_path"]
	pattern        = args["pattern"]
	first_video    = args["first_video"]
	last_video     = args["last_video"]
	predictor_path = args["predictor_path"]

	if not is_dir(videos_path):
		print('Invalid path to videos directory')
		return

	if not isinstance(output_path, str):
		print('Invalid path to output directory')
		return

	extract_to_npy(videos_path, output_path, predictor_path, pattern, first_video, last_video)


if __name__ == '__main__':
	main()
