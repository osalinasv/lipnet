from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers import Input
from keras import backend as K


class LipNet(object):
    CONV_ACTIVATION = 'relu'
    CONV_KERNEL = 'he_normal'

    def __init__(self, width=100, height=50, channels=3, frame_rate=30, max_string=32, output_size=28):
        self.width = width
        self.height = height
        self.channels = channels
        self.frame_rate = frame_rate
        self.max_string = max_string
        self.output_size = output_size

    def create(self):
        self.input = Input(shape=self.get_input_shape(), dtype='float32', name='input')

    def get_input_shape(self):
        if K.image_data_format() == 'channels_first':
            return self.channels, self.frame_rate, self.width, self.height
        else:
            return self.frame_rate, self.width, self.height, self.channels

    def create_conv(self, filters, kernel_size, shape, name: str) -> Conv3D:
        Conv3D(64, (3, 5, 5))
        return Conv3D(filters, (3, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv2')
