import os
import multiprocessing
import numpy as np
import pickle
from keras.callbacks import Callback
from lipnext.helpers.threadsafe import threadsafe_generator
from scipy import ndimage
from keras import backend as K
import sys
sys.path.append("..\\..")
from common.files import read_subfolders


class BatchGenerator(Callback):


    def __init__(self, *, dataset_path: str, minibatch_size: int, frame_count: int, image_channels: int,
                 image_height: int, image_width: int, max_string: int):
        self.data_path = os.path.realpath(dataset_path)

        self.train_path = os.path.join(self.data_path, 'train')
        self.val_path = os.path.join(self.data_path, 'val')
        self.align_path = os.path.join(self.data_path, 'align')

        self.minibatch_size = minibatch_size
        self.frame_count = frame_count
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width
        self.max_string = max_string

        self.cur_train_index = multiprocessing.Value('i', 0)
        self.cur_val_index = multiprocessing.Value('i', 0)

        self.shared_train_epoch = multiprocessing.Value('i', -1)
        self.process_train_index = -1
        self.process_val_index = -1

        self.steps_per_epoch = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps

        self.val_list = []
        self.train_list = []

        self.build_dataset()

    def get_cache_path(self):
        return self.data_path.rstrip('/') + '.cache'

    def build_dataset(self):
        cache_path = self.get_cache_path()

        if os.path.isfile(self.get_cache_path()):
            print("\nLoading dataset list from cache...")
            with open (self.get_cache_path(), 'rb') as fp:
                self.train_list, self.val_list, self.align_hash = pickle.load(fp)
        else:
            print("\nEnumerating dataset list from disk...")
            self.train_list = read_subfolders(self.train_path)
            self.val_list = read_subfolders(self.val_path)
            self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
            with open(self.get_cache_path(), 'wb') as fp:
                pickle.dump((self.train_list, self.val_list, self.align_hash), fp)
        print(cache_path)

    def enumerate_align_hash(self, video_list):
        align_hash = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split('\\')[-1]
            align_path = os.path.join(self.align_path, video_id)+".align"
            align_hash[video_id] = Align(self.absolute_max_string_len).from_file(align_path)
        return align_hash

    def read_dataset(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.val_list

        X_data_path = self.get_sublist(video_list, index, size)
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        for path in X_data_path:
            frames = Frames()
            align = self.align_hash[path.split('\\')[-1]]
            X_data.append(frames.get_data(path))
            Y_data.append(align.padded_label)
            label_length.append(align.label_length)  # CHANGED [A] -> A, CHECK!
            # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            input_length.append(
                frames.length)  # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            source_str.append(align.sentence)  # CHANGED [A] -> A, CHECK!

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(
            np.float32) / 255  # Normalize image data to [0,1], TODO: mean normalization over training data

        inputs = {'the_input': X_data,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        return (inputs, outputs)

    def get_sublist(self, l, index, size):
        ret = l[index:index + size]
        while size - len(ret) > 0:
            ret += l[0:size - len(ret)]
        return ret

    @threadsafe_generator
    def train_generator(self):
        while True:
            with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
                train_index = self.cur_train_index.value
                self.cur_train_index.value += self.minibatch_size

                if train_index >= self.steps_per_epoch * self.minibatch_size:
                    train_index = 0
                    self.shared_train_epoch.value += 1
                    self.cur_train_index.value = self.minibatch_size

                if self.shared_train_epoch.value < 0:
                    self.shared_train_epoch.value += 1

                # Shared index overflow
                if self.cur_train_index.value >= self.training_size:
                    self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size

            batch = self.read_dataset(train_index, self.minibatch_size, train=True)
            yield batch

    @threadsafe_generator
    def val_generator(self):
        while True:
            with self.cur_val_index.get_lock():
                val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size

                if self.cur_val_index.value >= self.validation_size:
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size

            batch = self.read_dataset(val_index, self.minibatch_size, train=False)
            yield batch

    def on_train_begin(self, logs={}):
        with self.cur_train_index.get_lock():
            self.cur_train_index.value = 0

        with self.cur_val_index.get_lock():
            self.cur_val_index.value = 0

    @property
    def training_size(self) -> int:
        return len(self.train_list)

    @property
    def default_training_steps(self) -> int:
        return self.training_size / self.minibatch_size

    @property
    def validation_size(self) -> int:
        return len(self.val_list)

    @property
    def default_validation_steps(self) -> int:
        return self.validation_size / self.minibatch_size

    def get_output_size(self):
        return 28

class Frames(object):
    def get_data(self, path):
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        frames = [ndimage.imread(frame_path) for frame_path in frames_path]
        return self.set_data(frames)

    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            frame = frame.swapaxes(0, 1)  # swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0, 2).swapaxes(0, 1)  # Add grayscale channel
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames)  # T x W x H x C
        if K.image_data_format() == 'channels_first':
            data_frames = np.rollaxis(data_frames, 3)  # C x T x W x H
        self.data = data_frames
        self.length = frames_n
        return self.data

class Align(object):
    def __init__(self, absolute_max_string_len=32):
        self.absolute_max_string_len = absolute_max_string_len

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
        self.build(align)
        return self

    def build(self, align):
        self.align = self.strip(align, ['sp','sil'])
        self.sentence = self.get_sentence(align)
        self.label = self.get_label(self.sentence)
        self.padded_label = self.get_padded_label(self.label)

    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        return " ".join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])

    def get_label(self, sentence):
        ret = []
        for char in sentence:
            if char >= 'a' and char <= 'z':
                ret.append(ord(char) - ord('a'))
            elif char == ' ':
                ret.append(26)
        return ret

    # Returns an array that is of size absolute_max_string_len. Fills the left spaces with -1 in case the len(label) is less than absolute_max_string_len.
    def get_padded_label(self, label):
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def label_length(self):
        return len(self.label)


if __name__ == '__main__':
    generator = BatchGenerator(
        dataset_path='path',
        minibatch_size=50,
        frame_count=30,
        image_channels=3,
        image_height=50,
        image_width=100,
        max_string=32
    )

    for idx, res in enumerate(generator.train_generator()):
        if idx >= 10:
            break

        print(res)
