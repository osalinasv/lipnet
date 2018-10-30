import argparse
import csv
import os
from typing import NamedTuple

import numpy as np
import skvideo.io
from colorama import Fore, init

import dlib
import env
from common.decode import create_decoder
from common.files import get_file_extension, get_files_in_dir, is_dir, is_file
from common.iters import chunks
from core.helpers.video import get_video_data_from_file, reshape_and_normalize_video_data
from core.model.lipnet import LipNet
from core.utils.visualization import visualize_video_subtitle
from preprocessing.extract_roi import extract_video_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
init(autoreset=True)


ROOT_PATH       = os.path.dirname(os.path.realpath(__file__))
DICTIONARY_PATH = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt'))


class PredictConfig(NamedTuple):
	weights:        str
	video_path:     str
	predictor_path: str
	frame_count:    int = env.FRAME_COUNT
	image_width:    int = env.IMAGE_WIDTH
	image_height:   int = env.IMAGE_HEIGHT
	image_channels: int = env.IMAGE_CHANNELS
	max_string:     int = env.MAX_STRING


def main():
	"""
	Entry point of the script for using a trained model for predicting videos.
	i.e: python predict.py -w data/res/2018-09-26-02-30/lipnet_065_1.96.hdf5 -v data/dataset_eval
	"""

	print(r'''
   __         __     ______   __   __     ______     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/     \/_/ 

  implemented by Omar Salinas
	''')

	ap = argparse.ArgumentParser()

	ap.add_argument('-v', '--video-path', required=True, help='Path to video file or batch directory to analize')
	ap.add_argument('-w', '--weights-path', required=True, help='Path to .hdf5 trained weights file')

	default_predictor = os.path.join(__file__, '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat')
	ap.add_argument("-pp", "--predictor-path", required=False, help="(Optional) Path to the predictor .dat file", default=default_predictor)

	args = vars(ap.parse_args())

	weights        = os.path.realpath(args['weights_path'])
	video          = os.path.realpath(args['video_path'])
	predictor_path = os.path.realpath(args["predictor_path"])

	if not is_file(weights) or get_file_extension(weights) != '.hdf5':
		print(Fore.RED + '\nERROR: Trained weights path is not a valid file')
		return

	if not is_file(video) and not is_dir(video):
		print(Fore.RED + '\nERROR: Path does not point to a video file nor to a directory')
		return

	if not is_file(predictor_path) or get_file_extension(predictor_path) != '.dat':
		print(Fore.RED + '\nERROR: Predictor path is not a valid file')
		return

	config = PredictConfig(weights, video, predictor_path)
	predict(config)


def predict(config: PredictConfig):
	print("\nPREDICTION\n")

	print('Loading weights at: {}'.format(config.weights))
	print('Using predictor at: {}'.format(config.predictor_path))

	print('\nMaking predictions...\n')

	lipnet = LipNet(config.frame_count, config.image_channels, config.image_height, config.image_width, config.max_string).compile_model().load_weights(config.weights)

	valid_paths   = []
	input_lengths = []
	predictions   = None

	elapsed_videos = 0
	video_paths = get_list_of_videos(config.video_path)

	for paths, lengths, y_pred in predict_batches(lipnet, video_paths, config.predictor_path):
		valid_paths   += paths
		input_lengths += lengths

		predictions = y_pred if predictions is None else np.append(predictions, y_pred, axis=0)

		y_pred_len = len(y_pred)
		elapsed_videos += y_pred_len

		print('Predicted batch of {} videos\t({} elapsed)'.format(y_pred_len, elapsed_videos))

	decoder = create_decoder(DICTIONARY_PATH)
	results = decode_predictions(predictions, input_lengths, decoder)

	print('\n\nRESULTS:\n')

	display   = query_yes_no('List all prediction outputs?', True)
	visualize = query_yes_no('Visualize as video captions?', False)

	print()

	save_csv = query_yes_no('Save prediction outputs to a .csv file?', True)

	if save_csv:
		csv_path = query_save_csv_path()
		write_results_to_csv(csv_path, valid_paths, results)
	
	if display or visualize:
		display_results(valid_paths, results, display, visualize)


def get_list_of_videos(path: str) -> [str]:
	path_is_file = is_file(path) and not is_dir(path)

	if path_is_file:
		print('Predicting for video at: {}'.format(path))
		video_paths = [path]
	else:
		print('Predicting batch at: {}'.format(path))
		video_paths = get_video_files_in_dir(path)

	return video_paths


def get_video_files_in_dir(path: str) -> [str]:
	return [f for ext in ['*.mpg', '*.npy'] for f in get_files_in_dir(path, ext)]


def get_video_data(path: str, detector, predictor) -> np.ndarray:
	if get_file_extension(path) == '.mpg':
		data = extract_video_data(path, detector, predictor, False)
		return reshape_and_normalize_video_data(data) if data is not None else None
	else:
		return get_video_data_from_file(path)


def get_entire_video_data(path: str) -> np.ndarray:
	if get_file_extension(path) == '.mpg':
		return np.swapaxes(skvideo.io.vread(path), 1, 2)
	else:
		return get_video_data_from_file(path)


def predict_batches(lipnet: LipNet, video_paths: [str], predictor_path: str):
	batch_size = env.BATCH_SIZE

	detector  = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	for paths in chunks(video_paths, batch_size):
		input_data = [(p, get_video_data(p, detector, predictor)) for p in paths]
		input_data = [x for x in input_data if x[1] is not None]

		if len(input_data) <= 0: continue

		valid_paths = [x[0] for x in input_data]

		x_data  = np.array([x[1] for x in input_data])
		lengths = [len(x) for x in x_data]

		y_pred = lipnet.predict(x_data)

		yield (valid_paths, lengths, y_pred)


def decode_predictions(y_pred: np.ndarray, input_lengths: list, decoder) -> list:
	input_lengths = np.array(input_lengths)
	return decoder.decode(y_pred, input_lengths)


def query_yes_no(query: str, default: bool=True) -> bool:
	prompt = '[Y/n]' if default else '[y/N]'
	inp = input(query + ' ' + prompt + ' ')

	return default if not inp else inp.lower()[0] == 'y'


def query_save_csv_path(default: str='output.csv'):
	path = input('Output CSV name (default is \'{}\'): '.format(default))

	if not path: path = default
	if not path.endswith('.csv'): path += '.csv'

	return os.path.realpath(path)


def display_results(valid_paths: list, results: list, display: bool=True, visualize: bool=False):
	if not display and not bool: return
	
	for p, r in zip(valid_paths, results):
		if display: print('\nVideo: {}\n    Result: {}'.format(p, r))

		if visualize:
			v = get_entire_video_data(p)
			visualize_video_subtitle(v, r)


def write_results_to_csv(path: str, valid_paths: list, results: list):
	already_exists = os.path.exists(path)

	with open(path, 'w') as f:
		writer = csv.writer(f)

		if not already_exists:
			writer.writerow(['file', 'prediction'])

		for p, r in zip(valid_paths, results):
			writer.writerow([p, r])


if __name__ == '__main__':
	main()
