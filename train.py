import argparse
import datetime
import os

from common.files import make_dir_if_not_exists


ROOT_PATH       = os.path.dirname(os.path.realpath(__file__))

DICTIONARY_PATH = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt'))
OUTPUT_DIR      = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'results'))
LOG_DIR         = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'logs'))


def train(run_name: str, epochs: int, frame_count: int, image_channels: int, image_height: int, image_width: int, max_string: int):
	from keras.callbacks import ModelCheckpoint, TensorBoard
	from lipnext.model.v2 import create_model, compile_model

	make_dir_if_not_exists(OUTPUT_DIR)
	make_dir_if_not_exists(LOG_DIR)

	print("\nStarted: Training...")
	print("Running: {}\n".format(run_name))

	CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, run_name)
	make_dir_if_not_exists(CHECKPOINT_DIR)

	tensorboard = TensorBoard(log_dir=os.path.join(LOG_DIR, run_name))
	checkpoint  = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "weights{epoch:04d}.hdf5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1, verbose=1, save_best_only=True)

	model = create_model(frame_count, image_channels, image_height, image_width, max_string)
	compile_model(model)

	# TODO: finish and replace generator & validation_data
	model.fit_generator(
		generator        = None,
		validation_data  = None,
		epochs           = epochs,
		verbose          = 1,
		workers          = 2,
		callbacks        = [checkpoint, tensorboard]
	)


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument('-e', '--epochs', required=False,
		help='(Optional) Number of epochs to run', type=int, default=5000)

	args = vars(ap.parse_args())

	epochs = args['epochs']

	name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	train(name, epochs, 75, 3, 50, 100, 32)


if __name__ == '__main__':
	main()
