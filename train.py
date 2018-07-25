import sys
sys.path.append('.')

import argparse
import datetime
import env
import os


ROOT_PATH       = os.path.dirname(os.path.realpath(__file__))

DICTIONARY_PATH = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt'))
OUTPUT_DIR      = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'results'))
LOG_DIR         = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'logs'))


# python train.py -d data/ordered -e 10
def train(run_name: str, dataset_path: str, epochs: int, frame_count: int, image_width: int, image_height: int, image_channels: int, max_string: int, batch_size: int, use_cache: bool):
	from common.files import make_dir_if_not_exists
	from keras.callbacks import ModelCheckpoint, TensorBoard
	from lipnext.model.v2 import Lipnext
	from lipnext.generators.dataset_generator import DatasetGenerator

	make_dir_if_not_exists(OUTPUT_DIR)
	make_dir_if_not_exists(LOG_DIR)

	print("\nTRAINING")
	print("Running: {}".format(run_name))

	CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, run_name)
	make_dir_if_not_exists(CHECKPOINT_DIR)

	tensorboard = TensorBoard(log_dir=os.path.join(LOG_DIR, run_name))
	checkpoint  = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "weights{epoch:04d}.hdf5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1, verbose=1, save_best_only=True)

	lipnext = Lipnext(frame_count, image_channels, image_height, image_width, max_string)
	lipnext.compile_model()

	datagen = DatasetGenerator(dataset_path, batch_size, max_string, use_cache)

	lipnext.model.fit_generator(
		generator       = datagen.train_generator,
		validation_data = datagen.val_generator,
		epochs          = epochs,
		shuffle         = True,
		verbose         = 1,
		max_queue_size  = 5,
		use_multiprocessing = True,
		workers         = 2,
		callbacks       = [checkpoint, tensorboard]
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
		help='Path to the structured dataset')

	ap.add_argument('-e', '--epochs', required=False,
		help='(Optional) Number of epochs to run', type=int, default=5000)

	ap.add_argument('-c', '--use-cache', required=False,
		help='(Optional) Load dataset from a cache file', type=bool, default=True)

	args = vars(ap.parse_args())

	dataset_path = args['dataset_path']
	epochs       = args['epochs']
	use_cache    = args['use_cache']

	name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	train(name, dataset_path, epochs, env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, env.IMAGE_CHANNELS, env.MAX_STRING, env.BATCH_SIZE, use_cache)


if __name__ == '__main__':
	main()
