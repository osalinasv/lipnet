import argparse
import os

import dlib
from colorama import Fore, Style, init

from common.files import get_file_extension, get_file_name, get_files_in_dir, is_dir, is_file, make_dir_if_not_exists
from preprocessing.extract_roi import video_to_frames


init(autoreset=True)


# set PYTHONPATH=%PYTHONPATH%;./
# python preprocessing\extract.py -v D:\GRID\ -o data\dataset -lv 840
def extract_to_npy(videos_path: str, output_path: str, predictor_path: str, pattern: str, first_video: int, last_video: int):
	print('\nEXTRACT\n')

	print('Searching for files in: {}'.format(videos_path))
	print('Using predictor at:     {}'.format(predictor_path))
	print('Match for:              {}\n'.format(pattern))

	videos_failed = []
	had_errors = False

	videos_failed_path = os.path.join(output_path, 'videos_failed.log')

	if is_file(videos_failed_path):
		with open(videos_failed_path, 'r+') as f:
			for line in f.readlines():
				videos_failed.append(line.rstrip())

	last_group_dir_name = ''
	video_count_per_group = 0

	detector  = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	for file_path in get_files_in_dir(videos_path, pattern):
		group_dir_name = os.path.basename(os.path.dirname(file_path))
		video_file_name = get_file_name(file_path)

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
			had_errors = True

			if video_count_per_group > 0:
				video_count_per_group -= 1
			else:
				video_count_per_group = 0

			with open(videos_failed_path, 'a+') as f:
				f.write(video_target_path + '\n')

	if had_errors:
		print(Fore.YELLOW + '\nExtraction finished with some errors')
	else:
		print(Fore.GREEN + '\nExtraction finished!')


def main():
	print(r'''
   __         __     ______   __   __     ______     __  __     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\_\_\_\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\_\/_  \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\   /\_\/\_\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/   \/_/\/_/     \/_/ 
	''')

	ap = argparse.ArgumentParser()

	ap.add_argument("-v",  "--videos-path", required=True, help="Path to videos directory")
	ap.add_argument("-o",  "--output-path", required=True, help="Path for the extracted frames")

	default_predictor = os.path.join(__file__, '..', '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat')
	ap.add_argument("-pp", "--predictor-path", required=False, help="(Optional) Path to the predictor .dat file", default=default_predictor)

	ap.add_argument("-p",  "--pattern", required=False, help="(Optional) File name pattern to match", default='*.mpg')
	ap.add_argument("-fv", "--first-video", required=False, help="(Optional) First video index extracted in each speaker (inclusive)", type=int, default=0)
	ap.add_argument("-lv", "--last-video",  required=False, help="(Optional) Last video index extracted in each speaker (exclusive)", type=int, default=1000)

	args = vars(ap.parse_args())

	videos_path    = os.path.realpath(args["videos_path"])
	output_path    = os.path.realpath(args["output_path"])
	pattern        = args["pattern"]
	first_video    = args["first_video"]
	last_video     = args["last_video"]
	predictor_path = os.path.realpath(args["predictor_path"])

	if not is_dir(videos_path):
		print(Fore.RED + 'ERROR: Invalid path to videos directory')
		return

	if not isinstance(output_path, str):
		print(Fore.RED + 'ERROR: Invalid path to output directory')
		return

	if not is_file(predictor_path) or get_file_extension(predictor_path) != '.dat':
		print(Fore.RED + '\nERROR: Predictor path is not a valid file')
		return

	if not isinstance(first_video, int) or first_video < 0:
		print(Fore.RED + '\nERROR: The first video index must be a valid positive integer')
		return

	if not isinstance(last_video, int) or last_video < first_video:
		print(Fore.RED + '\nERROR: The last video index must be a valid positive integer greater than the first video index')
		return

	extract_to_npy(videos_path, output_path, predictor_path, pattern, first_video, last_video)


if __name__ == '__main__':
	main()
