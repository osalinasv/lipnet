import numpy as np
import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w') # Patch to remove "Using TensorFlow backend" output
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator

import env

def get_video_data_from_file(path: str) -> np.ndarray:
	video_data = np.load(path) # T x H x W x C
	return reshape_and_normalize_video_data(video_data)


def reshape_and_normalize_video_data(video_data: np.ndarray) -> np.ndarray:
	return normalize_video_data(reshape_video_data(video_data))


def reshape_video_data(video_data: np.ndarray) -> np.ndarray:
	reshaped_video_data = video_data.swapaxes(1, 2) # T x W x H x C

	if len(reshaped_video_data) < 4:
		reshaped_video_data = reshaped_video_data.swapaxes(1, 3).swapaxes(1, 2) # Add grayscale channel

	if k.image_data_format() == 'channels_first':
		reshaped_video_data = np.rollaxis(reshaped_video_data, 3) # C x T x W x H

	return reshaped_video_data


def normalize_video_data(video_data: np.ndarray) -> np.ndarray:
	return video_data.astype(np.float32) / 255

def standardize(videos):
		print("Original shape of videos: {}".format(videos.shape))
		all_images = get_images_from_videos(videos)
		print("Shape of images: {}".format(all_images.shape))
		image_dg = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
		image_dg.fit(all_images)
		image_dg.standardize(all_images)
		videos = get_videos_from_images(all_images)
		print("Shape of videos: {}".format(videos.shape))
		return videos


def get_images_from_videos(videos):
		images = np.empty([0, 100, 50, 3])
		for video in videos:
				images = np.concatenate((images, video), axis=0)

		return images

def get_videos_from_images(images):
		total_images = np.size(images,0)
		total_videos = total_images / env.FRAME_COUNT
		s = 1
		videos = []
		while s <= total_videos:
				images_chunk = images[:env.FRAME_COUNT * s]
				print(np.array(videos).shape)
				videos.append(images_chunk)
				s += 1
		return np.array(videos)



