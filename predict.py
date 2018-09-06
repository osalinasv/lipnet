import argparse
import csv
import dlib
import env
import numpy as np
import os
import skvideo.io

from colorama import init, Fore
from common.files import is_dir, is_file, get_file_extension, get_files_in_dir, walk_level
from core.helpers.video import get_video_data_from_file, reshape_and_normalize_video_data
from preprocessing.extract_roi import extract_video_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
init(autoreset=True)


ROOT_PATH          = os.path.dirname(os.path.realpath(__file__))
DICTIONARY_PATH    = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt'))
DECODER_GREEDY     = True
DECODER_BEAM_WIDTH = 200


# set PYTHONPATH=%PYTHONPATH%;./
# python predict.py -w data\results\2018-08-28-00-04-11\w_0107_2.15.hdf5 -v data/dataset_eval
# bin blue at f two now
def predict(weights_file_path: str, video_path: str, predictor_path: str, frame_count: int, image_width: int, image_height: int, image_channels: int, max_string: int):
	from core.decoding.decoder import Decoder
	from core.decoding.spell import Spell
	from core.model.lipnext import Lipnext
	from core.utils.labels import labels_to_text
	from core.utils.visualization import visualize_video_subtitle
	
	
	print("\nPREDICTION\n")

	video_path_is_file = is_file(video_path) and not is_dir(video_path)

	if video_path_is_file:
		print('Predicting for video at: {}'.format(video_path))
		video_paths = [video_path]
	else:
		print('Predicting batch at:     {}'.format(video_path))
		video_paths = get_video_files_in_dir(video_path)

	print('Loading weights at:      {}'.format(weights_file_path))
	print('Using predictor at:      {}\n'.format(predictor_path))

	detector  = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	print('\nExtracting input video data...')

	input_data = list(map(lambda x: (x, path_to_video_data(x, detector, predictor)), video_paths))
	input_data = list(filter(lambda x: x[1] is not None, input_data))

	if len(input_data) <= 0:
		print(Fore.RED + '\nNo valid video files were found, exiting.')
		return

	print('\nMaking predictions...')

	lipnext = Lipnext(frame_count, image_channels, image_height, image_width, max_string)
	lipnext.compile_model()

	lipnext.model.load_weights(weights_file_path)

	x_data = np.array([x[1] for x in input_data])
	y_pred = None

	x_data_len = len(x_data)
	batch_size = env.BATCH_SIZE
	batch_iterations = int(np.ceil(x_data_len / batch_size))

	elapsed_videos = 0

	for idx in range(0, batch_iterations):
		split_start = idx * batch_size
		split_end   = split_start + batch_size

		if split_end > x_data_len:
			split_end = x_data_len

		elapsed_videos += split_end - split_start
		videos_batch = x_data[split_start:split_end]

		if y_pred is None:
			y_pred = lipnext.predict(videos_batch)
		else:
			y_pred = np.append(y_pred, lipnext.predict(videos_batch), axis=0)

		print('Predicted [{}/{}] videos ({}/{})'.format(elapsed_videos, x_data_len, idx + 1, batch_iterations))

	spell   = Spell(DICTIONARY_PATH)
	decoder = Decoder(greedy=DECODER_GREEDY, beam_width=DECODER_BEAM_WIDTH, postprocessors=[labels_to_text, spell.sentence])

	input_length = np.array([len(x) for x in x_data])
	results = decoder.decode(y_pred, input_length)

	print('\n\nRESULTS:\n')

	display_input  = input('Display outputs in console [Y/n]? ')
	display_videos = True if not display_input or display_input.lower()[0] == 'y' else False

	visualize_input  = input('Visualize results as video captions [y/N]? ')
	visualize_videos = visualize_input and visualize_input.lower()[0] == 'y'

	print()

	save_csv_input = input('Save outputs to CSV [y/N]? ')
	save_csv = save_csv_input and save_csv_input.lower()[0] == 'y'

	if save_csv:
		output_csv_path = input('Output CSV name (default is \'output\'): ')

		if not output_csv_path:
			output_csv_path = 'output.csv'

		if not output_csv_path.endswith('.csv'):
			output_csv_path += '.csv'

		output_csv_path = os.path.realpath(output_csv_path)

	if display_videos or visualize_videos:
		for (i, v), r in zip(input_data, results):
			if (display_videos):
				print('\nVideo: {}\n    Result: {}'.format(i, r))

			if visualize_videos:
				visualize_video_subtitle(v, r)

	if save_csv:
		output_csv_already_existed = os.path.exists(output_csv_path)

		with open(output_csv_path, 'w') as f:
			writer = csv.writer(f)

			if not output_csv_already_existed:
				writer.writerow(['file', 'prediction'])

			for (i, _), r in zip(input_data, results):
				writer.writerow([i, r])


def get_video_files_in_dir(path: str) -> [str]:
	return [f for ext in ['*.mpg', '*.npy'] for f in get_files_in_dir(path, ext)]


def path_to_video_data(path: str, detector, predictor) -> np.ndarray:
	if get_file_extension(path) == '.mpg':
		data = extract_video_data(path, detector, predictor)

		if data is not None:
			data = reshape_and_normalize_video_data(data)

		return data
	else:
		return get_video_data_from_file(path)


def main():
	print(r'''
   __         __     ______   __   __     ______     __  __     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\_\_\_\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\_\/_  \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\   /\_\/\_\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/   \/_/\/_/     \/_/ 
	''')

	ap = argparse.ArgumentParser()

	ap.add_argument('-v', '--video-path', required=True,
		help='Path to video file or batch directory to analize')

	ap.add_argument('-w', '--weights-path', required=True,
		help='Path to .hdf5 trained weights file')

	DEFAULT_PREDICTOR = os.path.join(__file__, '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat')

	ap.add_argument("-pp", "--predictor-path", required=False,
		help="(Optional) Path to the predictor .dat file", default=DEFAULT_PREDICTOR)

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
	
	predict(weights, video, predictor_path, env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, env.IMAGE_CHANNELS, env.MAX_STRING)


if __name__ == '__main__':
	main()
