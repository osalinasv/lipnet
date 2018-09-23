import os
from typing import Optional

import cv2
import numpy as np
import skvideo.io
from colorama import Back, Fore, init
from imutils import face_utils
from progress.bar import ShadyBar

import env
from common.files import get_file_name


init(autoreset=True)


FRAME_SHAPE = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH, env.IMAGE_CHANNELS)
IMAGE_SIZE  = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH)
ERROR_LOG   = Back.RED + Fore.BLACK + 'ERROR: '


def video_to_frames(video_path: str, output_path: str, detector, predictor) -> bool:
	video_path = os.path.realpath(video_path)
	video_data = extract_video_data(video_path, detector, predictor)

	if video_data is None:
		return False
	else:
		output_path = os.path.realpath(output_path)
		np.save(output_path, video_data)
		return True


def extract_video_data(path: str, detector, predictor, verbose: bool=True) -> Optional[np.ndarray]:
	if verbose: print('\n{}'.format(path))
	
	video_data     = skvideo.io.vread(path)
	video_data_len = len(video_data)

	if video_data_len != env.FRAME_COUNT:
		print(ERROR_LOG + 'Wrong number of frames: {}'.format(video_data_len))
		return None

	mouth_data = []
	bar = ShadyBar(get_file_name(path) + '.npy', max=video_data_len, suffix='%(percent)d%% [%(elapsed_td)s]') if verbose else None

	for i, f in enumerate(video_data):
		c = extract_mouth_on_frame(f, detector, predictor, i)
		if c is None: return None
		mouth_data.append(c)

		if verbose and bar: bar.next()

	mouth_data = np.array(mouth_data)
	if verbose and bar: bar.finish()

	return mouth_data


def extract_mouth_on_frame(frame: np.ndarray, detector, predictor, idx: int) -> Optional[np.ndarray]:
	m_points = extract_mouth_points(frame, detector, predictor)

	if m_points is None:
		print('\n' + ERROR_LOG + 'No ROI found at frame {}'.format(idx))
		return None

	m_center   = get_mouth_points_center(m_points)
	s_m_center = swap_center_axis(m_center)

	crop = crop_image(frame, s_m_center, IMAGE_SIZE)

	if crop.shape != FRAME_SHAPE:
		print('\n' + ERROR_LOG + 'Wrong shape {} at frame {}'.format(crop.shape, idx))
		return None

	return crop


def crop_image(image: np.ndarray, center: tuple, size: tuple) -> np.ndarray:
	start  = tuple(a - b // 2 for a, b in zip(center, size))
	end    = tuple(a + b for a, b in zip(start, size))
	slices = tuple(slice(a, b) for a, b in zip(start, end))

	return image[slices]


def swap_center_axis(t: np.ndarray) -> tuple:
	return t[1], t[0]


def get_mouth_points_center(mouth_points: np.ndarray) -> np.ndarray:
	mouth_centroid = np.mean(mouth_points[:, -2:], axis=0, dtype=int)
	return mouth_centroid


def extract_mouth_points(frame: np.ndarray, detector, predictor) -> Optional[np.ndarray]:
	gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detected = detector(gray, 1)

	if len(detected) <= 0:
		return None

	shape     = face_utils.shape_to_np(predictor(gray, detected[0]))
	_, (i, j) = list(face_utils.FACIAL_LANDMARKS_IDXS.items())[0]

	return np.array([shape[i:j]][0])
