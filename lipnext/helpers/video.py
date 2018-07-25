import numpy as np

from keras import backend as K


def get_video_data_from_file(path: str) -> np.ndarray:
	video_data = np.load(path) # T x H x W x C
	return reshape_and_normalize_video_data(video_data)


def reshape_and_normalize_video_data(video_data: np.ndarray) -> np.ndarray:
	return normalize_video_data(reshape_video_data(video_data))


def reshape_video_data(video_data: np.ndarray) -> np.ndarray:
	reshaped_video_data = np.array([reshape_video_frame(frame) for frame in video_data]) # T x W x H x C

	if K.image_data_format() == 'channels_first':
		reshaped_video_data = np.rollaxis(reshaped_video_data, 3) # C x T x W x H

	return reshaped_video_data


def reshape_video_frame(frame: np.ndarray) -> np.ndarray:
	frame = frame.swapaxes(0, 1) # swap width and height to form format W x H x C

	if len(frame.shape) < 3:
		frame = np.array([frame]).swapaxes(0, 2).swapaxes(0, 1)  # Add grayscale channel

	return frame


def normalize_video_data(video_data: np.ndarray) -> np.ndarray:
	return video_data.astype(np.float32) / 255
