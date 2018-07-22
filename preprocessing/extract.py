# TODO: remove this file

import argparse
import os
import shutil

from preprocessing.extractor.extract_roi_frames import video_to_frames
from common.files import is_dir, is_file, get_files_in_dir, make_dir_if_not_exists


def extract(videos_path: str, pattern: str, output_path: str, predictor_path: str, first_video: int, last_video: int):
	videos_path = os.path.realpath(videos_path)
	output_path = os.path.realpath(output_path)
	predictor_path = os.path.realpath(predictor_path)

	print('\nEXTRACT\n')
	print('Searching for files in: {}\nMatch for: {}\n'.format(videos_path, pattern))

	# Read what videos already succeeded
	videos_already_made = []
	VIDEOS_SUCCESS_PATH = os.path.join(output_path, 'videos_success.txt')

	if is_file(VIDEOS_SUCCESS_PATH):
		f = open(os.path.join(output_path, "videos_success.txt"), "r+")
		f_lines = f.readlines()

		for l in f_lines:
			videos_already_made.append(l.rstrip()) # remove line break

	last_group_dir = ''
	count_in_video = 0

	for file_path in get_files_in_dir(videos_path, pattern):
		group_dir = os.path.basename(os.path.dirname(file_path))
		video_dir = os.path.splitext(os.path.basename(file_path))[0]

		if group_dir == last_group_dir:
			count_in_video += 1
		else:
			count_in_video = 0

		last_group_dir = group_dir

		if count_in_video > last_video or count_in_video < first_video:
			continue

		video_full_dir = os.path.join(group_dir, video_dir)

		vid_cutouts_target_dir = os.path.join(output_path, video_full_dir)

		# if the video is already made the
		if video_full_dir in videos_already_made:
			print("Video was previously extracted " + video_full_dir + "\n")
			continue

		make_dir_if_not_exists(vid_cutouts_target_dir)

		if not video_to_frames(file_path, vid_cutouts_target_dir, predictor_path):
			count_in_video -= 1

			if count_in_video < 0:
				count_in_video = 0

			shutil.rmtree(vid_cutouts_target_dir)
			f = open(os.path.join(output_path, "videos_fail.txt"), "a+")
			f.write(video_full_dir + "\n")

		else:
			f = open(os.path.join(output_path, "videos_success.txt"), "a+")
			f.write(video_full_dir + "\n")

	print('Finished extraction successfully\n')


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument("-v",  "--videos-path", required=True,
		help="Path to videos directory")

	ap.add_argument("-o",  "--output-path", required=True, 
		help="Path for the extracted frames")

	ap.add_argument("-p",  "--pattern", required=False,
		help="(Optional) File name pattern to match", default='*.mpg')

	ap.add_argument("-fv", "--first-video", required=False,
		help="(Optional) First video extracted in each speaker inclusive", type=int, default=0)

	ap.add_argument("-lv", "--last-video",  required=False,
		help="(Optional) Lasst video extracted in each speaker inclusive", type=int, default=1001)

	DEFAULT_PREDICTOR = os.path.join(__file__, '..', '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat')

	ap.add_argument("-pp", "--predictor-path", required=False,
		help="(Optional) Path to the predictor .dat file", default=DEFAULT_PREDICTOR)

	args = vars(ap.parse_args())

	videos_path = args["videos_path"]
	output_path = args["output_path"]
	pattern = args["pattern"]
	first_video = args["first_video"]
	last_video = args["last_video"]
	predictor_path = args["predictor_path"]

	if not is_dir(videos_path):
		print('Invalid path to videos directory')
		return

	if not isinstance(output_path, str):
		print('Invalid path to output directory')
		return

	extract(videos_path, pattern, output_path, predictor_path, first_video, last_video)


if __name__ == '__main__':
	main()
