import cv2
import dlib
import env
import numpy as np
import os
import skvideo.io

from colorama import init, Back, Fore, Style
from imutils import face_utils
from progress.bar import ShadyBar
from scipy.misc import imresize


init(autoreset=True)


def video_to_frames(video_path: str, output_path: str, detector, predictor) -> bool:
	video_path  = os.path.realpath(video_path)
	output_path = os.path.realpath(output_path)

	video_data  = extract_video_data(video_path, detector, predictor)

	if video_data is None:
		return False
	else:
		np.save(output_path, video_data)
		return True


def extract_video_data(video_path: str, detector, predictor) -> np.ndarray:
	print('\nProcessing: {}'.format(video_path))

	frames_array = skvideo.io.vread(video_path)

	if len(frames_array) != env.FRAME_COUNT:
		print(Back.RED + Fore.BLACK + 'ERROR: Video {} does not match the frame count specified'.format(video_path))
		return None

	mouth_frames_array = []

	# Progress bar
	bar = ShadyBar(os.path.basename(video_path), max=len(frames_array), suffix='%(percent)d%% [%(elapsed_td)s]')

	for i, frame in enumerate(frames_array):
		mouth_frame = extract_mouth(frame, detector, predictor)

		if mouth_frame is None:
			print(Back.RED + Fore.BLACK + 'ERROR: Could not find ROI at frame {} of video {}'.format(i, video_path))
			return None

		mouth_frames_array.append(mouth_frame)
		bar.next()
	
	bar.finish()
	return np.array(mouth_frames_array)


def extract_mouth(frame, detector, predictor):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	for i, rect in enumerate(detector(gray, 1)):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# Obtain the mouth landmark at index 0
		# See: https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py
		(_, (i, j)) = list(face_utils.FACIAL_LANDMARKS_IDXS.items())[0]

		# Extract the ROI of the face region as a separate image
		np_mouth_points = np.array([shape[i:j]])

		return crop_mouth_region(np_mouth_points[0], frame)


def crop_mouth_region(np_mouth_points, frame):
	normalize_ratio = None
	mouth_centroid  = np.mean(np_mouth_points[:, -2:], axis=0)

	if normalize_ratio is None:
		mouth_left  = np.min(np_mouth_points[:, :-1]) * (1.0 - env.HORIZONTAL_PAD)
		mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + env.HORIZONTAL_PAD)

		normalize_ratio = env.IMAGE_WIDTH / float(mouth_right - mouth_left)

	new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
	resized_img   = imresize(frame, new_img_shape)

	mouth_centroid_norm = mouth_centroid * normalize_ratio

	mouth_l = int(mouth_centroid_norm[0] - env.IMAGE_WIDTH / 2)
	mouth_r = int(mouth_centroid_norm[0] + env.IMAGE_WIDTH / 2)
	mouth_t = int(mouth_centroid_norm[1] - env.IMAGE_HEIGHT / 2)
	mouth_b = int(mouth_centroid_norm[1] + env.IMAGE_HEIGHT / 2)

	diff_width = mouth_r - mouth_l

	if diff_width > env.IMAGE_WIDTH:
		mouth_r += env.IMAGE_WIDTH - diff_width

	diff_height = mouth_b - mouth_t

	if diff_height > env.IMAGE_HEIGHT:
		mouth_b += env.IMAGE_HEIGHT - diff_height

	mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

	return mouth_crop_image
