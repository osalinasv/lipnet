import argparse
import datetime
import env
import os

from colorama import init, Fore
from common.files import is_dir


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
init(autoreset=True)


ROOT_PATH  = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'results'))
LOG_DIR    = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'logs'))


# python train.py -d data/dataset -a D:/GRID/align/ -e 10
def train(run_name: str, dataset_path: str, aligns_path: str, epochs: int, frame_count: int, image_width: int, image_height: int, image_channels: int, max_string: int, batch_size: int, val_split: float, use_cache: bool):
	from common.files import make_dir_if_not_exists
	from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
	from lipnext.model.v4 import Lipnext
	from lipnext.generators.dataset_generator import DatasetGenerator


	print("\nTRAINING\n")

	print("Running: {}\n".format(run_name))

	print('For dataset at: {}'.format(dataset_path))
	print('With aligns at: {}'.format(aligns_path))

	make_dir_if_not_exists(OUTPUT_DIR)
	make_dir_if_not_exists(LOG_DIR)

	checkpoint_dir = os.path.join(OUTPUT_DIR, run_name)
	make_dir_if_not_exists(checkpoint_dir)

	run_log_dir = os.path.join(LOG_DIR, run_name)
	csv_log_dir = os.path.join(run_log_dir, '{}_train.csv'.format(run_name))

	tensorboard = TensorBoard(log_dir=run_log_dir)
	csv_logger  = CSVLogger(csv_log_dir, separator=',', append=True)
	checkpoint  = ModelCheckpoint(os.path.join(checkpoint_dir, "w_{epoch:04d}_{val_loss:.2f}.hdf5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1, verbose=1, save_best_only=True)

	lipnext = Lipnext(frame_count, image_channels, image_height, image_width, max_string)
	lipnext.compile_model()

	datagen = DatasetGenerator(dataset_path, aligns_path, batch_size, max_string, val_split, use_cache)

	lipnext.model.fit_generator(
		generator       = datagen.train_generator,
		validation_data = datagen.val_generator,
		epochs          = epochs,
		shuffle         = True,
		verbose         = 1,
		max_queue_size  = 10,
		use_multiprocessing = True,
		workers         = 2,
		callbacks       = [checkpoint, tensorboard, csv_logger]
	)


def main():
	print(r'''
   __         __     ______   __   __     ______     __  __     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\_\_\_\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\_\/_  \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\   /\_\/\_\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/   \/_/\/_/     \/_/ 
	''')

	ap = argparse.ArgumentParser()

	ap.add_argument('-d', '--dataset-path', required=True,
		help='Path to the dataset root directory')

	ap.add_argument('-a', '--aligns-path', required=True,
		help='Path to the directory containing all align files')

	ap.add_argument('-e', '--epochs', required=False,
		help='(Optional) Number of epochs to run', type=int, default=5000)

	ap.add_argument('-c', '--use-cache', required=False,
		help='(Optional) Load dataset from a cache file', type=bool, default=True)

	args = vars(ap.parse_args())

	dataset_path = os.path.realpath(args['dataset_path'])
	aligns_path  = os.path.realpath(args['aligns_path'])
	epochs       = args['epochs']
	use_cache    = args['use_cache']

	if not is_dir(dataset_path):
		print(Fore.RED + '\nERROR: The dataset path is not a directory')
		return

	if not is_dir(aligns_path):
		print(Fore.RED + '\nERROR: The aligns path is not a directory')
		return

	if not isinstance(epochs, int) or epochs <= 0:
		print(Fore.RED + '\nERROR: The number of epochs must be a valid integer greater than zero')
		return

	name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	
	train(name, dataset_path, aligns_path, epochs, env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, env.IMAGE_CHANNELS, env.MAX_STRING, env.BATCH_SIZE, env.VAL_SPLIT, use_cache)


if __name__ == '__main__':
	main()
