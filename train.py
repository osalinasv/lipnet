import argparse
import datetime
import os
from typing import NamedTuple

from colorama import Fore, init
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

import env
from common.decode import create_decoder
from common.files import is_dir, make_dir_if_not_exists
from core.callbacks.error_rates import ErrorRates
from core.generators.dataset_generator import DatasetGenerator
from core.model.lipnext import LipNext


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
init(autoreset=True)


ROOT_PATH  = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'results'))
LOG_DIR    = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'logs'))

DICTIONARY_PATH = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt'))


class TrainingConfig(NamedTuple):
	dataset_path:   str
	aligns_path:    str
	epochs:         int = 1
	frame_count:    int = env.FRAME_COUNT
	image_width:    int = env.IMAGE_WIDTH
	image_height:   int = env.IMAGE_HEIGHT
	image_channels: int = env.IMAGE_CHANNELS
	max_string:     int = env.MAX_STRING
	batch_size:     int = env.BATCH_SIZE
	val_split:    float = env.VAL_SPLIT
	use_cache:     bool = True


def create_callbacks(run_name: str, lipnext: LipNext, datagen: DatasetGenerator) -> list:
	# Tensorboard
	run_log_dir = os.path.join(LOG_DIR, run_name)
	tensorboard = TensorBoard(log_dir=run_log_dir)

	# Training logger
	csv_log_dir = os.path.join(run_log_dir, '{}_train.csv'.format(run_name))
	csv_logger  = CSVLogger(csv_log_dir, separator=',', append=True)

	# Model checkpoint saver
	checkpoint_dir = os.path.join(OUTPUT_DIR, run_name)
	make_dir_if_not_exists(checkpoint_dir)

	checkpoint_template = os.path.join(checkpoint_dir, "w_{epoch:04d}_{val_loss:.2f}.hdf5")
	checkpoint = ModelCheckpoint(checkpoint_template, monitor='val_loss', save_weights_only=True, mode='auto', period=1, verbose=1)

	# WER/CER Error rate calculator
	error_rate_log_dir = os.path.join(run_log_dir, '{}_error_rate.csv'.format(run_name))

	decoder = create_decoder(DICTIONARY_PATH, False)
	error_rates = ErrorRates(error_rate_log_dir, lipnext, datagen.val_generator, decoder)

	return [checkpoint, tensorboard, csv_logger, error_rates]


def train(run_name: str, config: TrainingConfig):
	print("\nTRAINING\n")

	print("Running: {}\n".format(run_name))

	print('For dataset at: {}'.format(config.dataset_path))
	print('With aligns at: {}'.format(config.aligns_path))

	make_dir_if_not_exists(OUTPUT_DIR)
	make_dir_if_not_exists(LOG_DIR)

	lipnext = LipNext(config.frame_count, config.image_channels, config.image_height, config.image_width, config.max_string).compile_model()

	datagen = DatasetGenerator(config.dataset_path, config.aligns_path, config.batch_size, config.max_string, config.val_split, config.use_cache)

	callbacks = create_callbacks(run_name, lipnext, datagen)

	print('\nStarting training...\n')

	lipnext.model.fit_generator(
		generator      =datagen.train_generator,
		validation_data=datagen.val_generator,
		epochs         =config.epochs,
		verbose        =1,
		shuffle        =True,
		max_queue_size =5,
		workers        =3,
		callbacks      =callbacks,
		use_multiprocessing=True
	)

	print('\nTraining completed')


def main():
	print(r'''
   __         __     ______   __   __     ______     __  __     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\_\_\_\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\_\/_  \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\   /\_\/\_\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/   \/_/\/_/     \/_/ 
	''')

	ap = argparse.ArgumentParser()

	ap.add_argument('-d', '--dataset-path', required=True, help='Path to the dataset root directory')
	ap.add_argument('-a', '--aligns-path', required=True, help='Path to the directory containing all align files')
	ap.add_argument('-e', '--epochs', required=False, help='(Optional) Number of epochs to run', type=int, default=1)

	args = vars(ap.parse_args())

	dataset_path = os.path.realpath(args['dataset_path'])
	aligns_path  = os.path.realpath(args['aligns_path'])
	epochs       = args['epochs']

	if not is_dir(dataset_path):
		print(Fore.RED + '\nERROR: The dataset path is not a directory')
		return

	if not is_dir(aligns_path):
		print(Fore.RED + '\nERROR: The aligns path is not a directory')
		return

	if not isinstance(epochs, int) or epochs <= 0:
		print(Fore.RED + '\nERROR: The number of epochs must be a valid integer greater than zero')
		return

	name   = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	config = TrainingConfig(dataset_path, aligns_path, epochs=epochs)

	train(name, config)


if __name__ == '__main__':
	main()
