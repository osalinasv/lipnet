from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers import Input
from keras.models import Model
from keras import backend as k


class LipNext(object):
    CONV_ACTIVATION = 'relu'
    CONV_KERNEL = 'he_normal'

    DROPOUT_RATE = 0.5

    GRU_KERNEL = 'Orthogonal'
    GRU_MERGE_MODE = 'concat'

    def __init__(self, width=100, height=50, channels=3, frame_rate=30, max_string=32, output_size=28):
        self.width = width
        self.height = height
        self.channels = channels
        self.frame_rate = frame_rate
        self.max_string = max_string
        self.output_size = output_size

        self.create()

    def create(self):
        self.input_layer = Input(shape=self.get_input_shape(), dtype='float32', name='input')

        self.zero_1 = self.create_zero('zero_1', self.input_layer)
        self.conv_1 = self.create_conv('conv_1', self.zero_1, 32)
        self.pool_1 = self.create_pool('pool_1', self.conv_1)
        self.drop_1 = self.create_drop(self.pool_1)

        self.zero_2 = self.create_zero('zero_2', self.drop_1)
        self.conv_2 = self.create_conv('conv_2', self.zero_2, 64)
        self.pool_2 = self.create_pool('pool_2', self.conv_2)
        self.drop_2 = self.create_drop(self.pool_2)

        self.zero_3 = self.create_zero('zero_3', self.drop_2, padding=(1, 1, 1))
        self.conv_3 = self.create_conv('conv_3', self.zero_3, 96, kernel_size=(3, 3, 3))
        self.pool_3 = self.create_pool('pool_3', self.conv_3)
        self.drop_3 = self.create_drop(self.pool_3)

        self.res = TimeDistributed(Flatten())(self.drop_3)

    def get_input_shape(self):
        if k.image_data_format() == 'channels_first':
            return self.channels, self.frame_rate, self.width, self.height
        else:
            return self.frame_rate, self.width, self.height, self.channels

    @staticmethod
    def create_zero(name: str, input_layer, padding=(1, 2, 2)) -> ZeroPadding3D:
        return ZeroPadding3D(padding=padding, name=name)(input_layer)

    @staticmethod
    def create_conv(name: str, input_layer, filters: int, kernel_size=(3, 5, 5)) -> Conv3D:
        return Conv3D(filters, kernel_size, strides=(1, 2, 2), activation=LipNext.CONV_ACTIVATION,
                      kernel_initializer=LipNext.CONV_KERNEL, name=name)(input_layer)

    @staticmethod
    def create_pool(name: str, input_layer) -> MaxPooling3D:
        return MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name=name)(input_layer)

    @staticmethod
    def create_drop(input_layer) -> Dropout:
        return Dropout(LipNext.DROPOUT_RATE)(input_layer)