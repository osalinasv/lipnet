import fnmatch
import os
import sys
import shutil
import argparse

from preprocessing.extractor.extract_roi_frames import video_to_frames
from common.files import make_dir_if_not_exists, get_files_in_dir


def extract(videos_path: str, pattern: str, output_path: str, predictor_path: str, first_video: int, last_video: int):
	"""
	Extracts the frames in all videos inside videos_path that match pattern

	Usage:
		python extract.py [videos_path] [pattern] [output_path]

		videos_path         Path to videos directory
		pattern             Filename pattern to match
		output_path         Path for the extracted frames

	Example:
		python extract.py data/dataset *.mpg data/target data/predictors/shape_predictor_68_face_landmarks.dat

	:param videos_path:
	:param pattern:
	:param output_path:
	:param predictor_path:
	:return:
	"""

	videos_path = os.path.realpath(videos_path)
	output_path = os.path.realpath(output_path)
	predictor_path = os.path.realpath(predictor_path)

	print('\nEXTRACT\n')
	print('Searching for files in: {}\nMatch for: {}'.format(videos_path, pattern))

	# Read what videos already success
	videos_already_made = []
	f = open(os.path.join(output_path, "videos_success.txt"), "r+")
	fLines = f.readlines()
	for l in fLines:
		videos_already_made.append(l.rstrip())  # remove line break

	last_group_dir = ""
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


if __name__ == '__main__':
	'''
		extract.py
			Extracts the frames in all videos inside videos_path that match pattern

		Usage:
			python extract.py [videos_path] [pattern] [output_path] [predictor_path]

			videos_path         Path to videos directory
			pattern             (Optional) Filename pattern to match
			output_path         Path for the extracted frames
			from_video          (Optional) First video extracted in each speaker inclusive
			to_video            (Optional) Last video extracted in each speaker inclusive
			predictor_path      (Optional) Path to the predictor .dat file

		Example:
			python extract.py -v data/dataset -p *.mpg -o data/target -fv 0 -lv 1000 -pp data/predictors/shape_predictor_68_face_landmarks.dat

	'''

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--videos-path", required=True, help="Path to videos directory")
	ap.add_argument("-p", "--pattern", required=False, help="(Optional) Filename pattern to match", default='*.mpg')
	ap.add_argument("-o", "--outputh-path", required=True, help="Path for the extracted frames")
	ap.add_argument("-fv", "--first-video", required=False, help="(Optional) First video extracted in each speaker inclusive", type=int, default=0)
	ap.add_argument("-lv", "--last-video", required=False, help="(Optional) Lasst video extracted in each speaker inclusive", type=int, default=1000)
	ap.add_argument("-pp", "--predictor-path", required=False, help="(Optional) Path to the predictor .dat file", default=os.path.join(__file__, '..', '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat'))

	args = vars(ap.parse_args())

	i_path = args["videos_path"]
	o_path = args["outputh_path"]
	pat = args["pattern"]
	first_video = args["first_video"]
	last_video = args["last_video"]
	p_path = args["predictor_path"]

	if i_path is None or o_path is None:
		print('Both input and output are required\n')
		exit()

	extract(i_path, pat, o_path, p_path,
			first_video=first_video, last_video=last_video)
