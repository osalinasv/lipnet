import datetime
import os

from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

from lipnext.model.v1 import LipNext
from lipnext.utils.spell import Spell

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CURRENT_PATH, '..')

DICTIONARY_PATH = os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt')
OUTPUT_DIR = os.path.join(ROOT_PATH, 'data', 'results')
LOG_DIR = os.path.join(ROOT_PATH, 'data', 'logs')


def train(*, run_name: str, start_epoch: int, stop_epoch: int):
    print("Started: Training...")
    print("Running: {}".format(run_name))

    lipnext = LipNext()
    lipnext.summary()
    lipnext.compile()

    spell = Spell(DICTIONARY_PATH)

    tensorboard = TensorBoard(log_dir=os.path.join(LOG_DIR, run_name))
    csv_logger = CSVLogger(os.path.join(LOG_DIR, "{}-{}.csv".format('training', run_name)), separator=',', append=True)
    checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, run_name, "weights{epoch:02d}.h5"), monitor='val_loss',
                                 save_weights_only=True, mode='auto', period=1)

    # TODO: set generator, validator, training_steps, validation_steps
    lipnext.train(
        generator=None,
        validator=None,
        start_epoch=start_epoch,
        stop_epoch=stop_epoch,
        training_steps=None,
        validation_steps=None,
        callbacks=[checkpoint, tensorboard, csv_logger]
    )


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    train(run_name=run_name, start_epoch=0, stop_epoch=1)
