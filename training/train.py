import datetime
import os

from common.files import make_dir


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH    = os.path.realpath(os.path.join(CURRENT_PATH, '..'))

DICTIONARY_PATH = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt'))
OUTPUT_DIR      = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'results'))
LOG_DIR         = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'logs'))


def train(run_name: str, epochs: int, frame_count: int, image_channels: int, image_height: int, image_width: int, max_string: int):
	from keras.callbacks import ModelCheckpoint, TensorBoard
	from lipnext.model.v2 import create_model, compile_model

	make_dir(OUTPUT_DIR)
	make_dir(LOG_DIR)

	print("\nStarted: Training...")
	print("Running: {}\n".format(run_name))

	model = create_model(frame_count, image_channels, image_height, image_width, max_string)
	model.summary()

	CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, run_name)
	make_dir(CHECKPOINT_DIR)

	tensorboard = TensorBoard(log_dir=os.path.join(LOG_DIR, run_name))
	checkpoint  = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "weights{epoch:04d}.h5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1)

	compile_model(model)

	model.fit_generator(
		generator        = None,
		validation_data  = None,
		epochs           = epochs,
		verbose          = 1,
		workers          = 2,
		callbacks        = [checkpoint, tensorboard]
	)


if __name__ == '__main__':
	name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

	print('CURRENT_PATH: {}'.format(CURRENT_PATH))
	print('ROOT_PATH: {}'.format(ROOT_PATH))
	print('DICTIONARY_PATH: {}'.format(DICTIONARY_PATH))
	print('OUTPUT_DIR: {}'.format(OUTPUT_DIR))
	print('LOG_DIR: {}'.format(LOG_DIR))

	# train(
	# 	run_name       = name,
	# 	start_epoch    = 0,
	# 	stop_epoch     = 5000,
	# 	frame_count    = 75,
	# 	image_channels = 3,
	# 	image_height   = 50,
	# 	image_width    = 100,
	# 	max_string     = 32,
	# )
