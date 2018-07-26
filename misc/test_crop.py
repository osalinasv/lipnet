import cv2
import dlib
import numpy as np
import operator
import os
import skvideo.io

from colorama import init, Back, Fore
from imutils import face_utils
from progress.bar import ShadyBar
from timeit import time
from matplotlib import pyplot as plt


init(autoreset=True)


FC = 75
IW = 100
IH = 50
IC = 3

VIDEO_SHAPE = (FC, IH, IW, IC)
FRAME_SHAPE = (IH, IW, IC)
ERROR_LOG = Back.RED + Fore.BLACK + 'ERROR: '


def extract_mouth_points(frame: np.ndarray, detector, predictor) -> np.ndarray:
	gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detected = detector(gray, 1)

	if len(detected) <= 0:
		return None

	shape = face_utils.shape_to_np(predictor(gray, detected[0]))
	_, (i, j) = list(face_utils.FACIAL_LANDMARKS_IDXS.items())[0]

	return np.array([shape[i:j]][0])


def get_mouth_points_center(mouth_points: np.ndarray) -> np.ndarray:
	mouth_centroid = np.mean(mouth_points[:, -2:], axis=0, dtype=int)
	return mouth_centroid


def swap_center_axis(t: tuple) -> tuple:
	return t[1], t[0]


def crop_image(image: np.ndarray, center: tuple, size: tuple) -> np.ndarray:
	start  = tuple(map(lambda a, b: a - b // 2, center, size))
	end    = tuple(map(operator.add, start, size))
	slices = tuple(map(slice, start, end))

	return image[slices]


def extract_mouth_on_frame(frame: np.ndarray, idx: int) -> np.ndarray:
	m_points   = extract_mouth_points(frame, detector, predictor)

	if m_points is None:
		print('\n' + ERROR_LOG + 'No ROI found at frame {}'.format(idx))
		return None

	m_center   = get_mouth_points_center(m_points)
	s_m_center = swap_center_axis(m_center)

	crop = crop_image(frame, s_m_center, (IH, IW))

	if crop.shape != FRAME_SHAPE:
		print('\n' + ERROR_LOG + 'Wrong shape {} at frame {}'.format(crop.shape, idx))
		return None

	return crop


def extract_video_data(path: str, detector, predictor) -> np.ndarray:
	video_data = skvideo.io.vread(path)
	video_data_len = len(video_data)

	if video_data_len != FC:
		print(ERROR_LOG + 'Wrong number of frames: {}'.format(video_data_len))
		return None

	mouth_data = []
	bar = ShadyBar(os.path.basename(path), max=video_data_len, suffix='%(percent)d%% [%(elapsed_td)s]')

	for i, f in enumerate(video_data):
		c = extract_mouth_on_frame(f, i)
		if c is None: return None
		mouth_data.append(c)

		bar.next()

	mouth_data = np.array(mouth_data)
	bar.finish()

	return mouth_data



detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.realpath('./data/predictors/shape_predictor_68_face_landmarks.dat'))

# D:/GRID/s1/bbaf2n.mpg  - good one
# D:/GRID/s22/brij8s.mpg - prev algorithm wrong reshaping (49, 100, 3)
# D:/GRID/s24/bbii9n.mpg - no ROI at frame 12

test_videos = [
	'D:/GRID/s1/bbaf2n.mpg',
	'D:/GRID/s22/brij8s.mpg',
	'D:/GRID/s24/bbii9n.mpg'
]

for i, p in enumerate(test_videos):
	print('\n{}\n'.format(p))
	start_time = time.time()

	data = extract_video_data(p, detector, predictor)

	if data is not None:
		print('shape:   {}'.format(data.shape))

	print('elapsed: {}'.format(time.time() - start_time))

	# if data is not None and i == 0:
	# 	writer = cv2.VideoWriter('tools/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (IW, IH))
	# 	for f in data:
	# 		writer.write(f)
		
	# 	writer.release()

	# 	plt.imshow(data[0])
	# 	plt.show()
