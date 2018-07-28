import argparse
import dlib
import env
import numpy as np
import os
import skvideo

from colorama import init, Fore
from common.files import is_file, get_file_extension


init(autoreset=True)


ROOT_PATH          = os.path.dirname(os.path.realpath(__file__))
DICTIONARY_PATH    = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt'))
DECODER_GREEDY     = True
DECODER_BEAM_WIDTH = 200


# set PYTHONPATH=%PYTHONPATH%;./
# python predict.py -w data\results\2018-07-27-22-22-14\w-0002-31.14.hdf5 -v D:\GRID\s34\sbwe6a.mpg
# set blue with e six again
def predict(weights_file_path: str, video_file_path: str, predictor_path: str, frame_count: int, image_width: int, image_height: int, image_channels: int, max_string: int):
	from lipnext.decoding.decoder import Decoder
	from lipnext.decoding.spell import Spell
	from lipnext.decoding.visualization import visualize_video_subtitle
	from lipnext.helpers.video import reshape_and_normalize_video_data
	from lipnext.model.v2 import Lipnext
	from lipnext.utils.labels import labels_to_text
	from preprocessing.extractor.extract_roi import extract_video_data

	print("\nPREDICTION\n")

	print('Predicting for video at: {}'.format(video_file_path))
	print('Loading weights at:      {}'.format(weights_file_path))
	print('Using predictor at:      {}'.format(predictor_path))

	detector  = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	video_data = extract_video_data(video_file_path, detector, predictor)
	video_data = reshape_and_normalize_video_data(video_data)

	print()

	lipnext = Lipnext(frame_count, image_channels, image_height, image_width, max_string)
	lipnext.compile_model()

	lipnext.model.load_weights(weights_file_path)

	x_data = np.array([video_data])
	y_pred = lipnext.predict(x_data)

	input_length = np.array([len(video_data)])

	spell   = Spell(DICTIONARY_PATH)
	decoder = Decoder(greedy=DECODER_GREEDY, beam_width=DECODER_BEAM_WIDTH,
		postprocessors=[labels_to_text, spell.sentence])

	result = decoder.decode(y_pred, input_length)[0]

	print('\ny_pred shape:   {}'.format(y_pred.shape))
	print('decoded result: {}'.format(result))

	print('\nPreparing visualization...')
	original_video_data = skvideo.io.vread(video_file_path)
	visualize_video_subtitle(original_video_data, result)


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
		help='Path to video file to analize')

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

	if not is_file(video):
		print(Fore.RED + '\nERROR: Video file path is not valid')
		return

	if not is_file(predictor_path) or get_file_extension(predictor_path) != '.dat':
		print(Fore.RED + '\nERROR: Predictor path is not a valid file')
		return
	
	predict(weights, video, predictor_path, env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, env.IMAGE_CHANNELS, env.MAX_STRING)


if __name__ == '__main__':
	main()
