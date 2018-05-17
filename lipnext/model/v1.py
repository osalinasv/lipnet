from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, Flatten, SpatialDropout3D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.models import Model
from keras import backend as k
from lipnext.model.layers import CTC
from keras.utils import plot_model


class LipNext(object):
    IMAGE_WIDTH = 100
    IMAGE_HEIGHT = 50
    IMAGE_CHANNELS = 3
    FRAME_RATE = 30
    MAX_STRING = 32
    OUTPUT_SIZE = 28

    INPUT_TYPE = 'float32'

    ZERO_PADDING = (1, 2, 2)

    CONV_ACTIVATION = 'relu'
    CONV_KERNEL_INIT = 'he_normal'
    CONV_KERNEL_SIZE = (3, 5, 5)
    CONV_STRIDES = (1, 2, 2)

    POOL_SIZE = (1, 2, 2)
    POOL_STRIDES = (1, 2, 2)

    DROPOUT_RATE = 0.5

    GRU_KERNEL_INIT = 'Orthogonal'
    GRU_MERGE_MODE = 'concat'

    def __init__(self, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, image_channels=IMAGE_CHANNELS,
                 frame_rate=FRAME_RATE, max_string=MAX_STRING, output_size=OUTPUT_SIZE):
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.frame_rate = frame_rate
        self.max_string = max_string
        self.output_size = output_size

        self.model: Model = None

        self.create()

    def create(self):
        input_layer = LipNext.create_input('input', shape=self.get_input_shape())

        zero_1 = LipNext.create_zero('zero_1', input_layer)
        conv_1 = LipNext.create_conv('conv_1', zero_1, 32)
        batc_1 = LipNext.create_batc('batc_1', conv_1)
        actv_1 = LipNext.create_actv('actv_1', batc_1)
        drop_1 = LipNext.create_drop('drop_1', actv_1)
        pool_1 = LipNext.create_pool('pool_1', drop_1)

        zero_2 = LipNext.create_zero('zero_2', pool_1)
        conv_2 = LipNext.create_conv('conv_2', zero_2, 64)
        batc_2 = LipNext.create_batc('batc_2', conv_2)
        actv_2 = LipNext.create_actv('actv_2', batc_2)
        drop_2 = LipNext.create_drop('drop_2', actv_2)
        pool_2 = LipNext.create_pool('pool_2', drop_2)

        zero_3 = LipNext.create_zero('zero_3', pool_2, padding=(1, 1, 1))
        conv_3 = LipNext.create_conv('conv_3', zero_3, 96, kernel_size=(3, 3, 3))
        batc_3 = LipNext.create_batc('batc_3', conv_3)
        actv_3 = LipNext.create_actv('actv_3', batc_3)
        drop_3 = LipNext.create_drop('drop_3', actv_3)
        pool_3 = LipNext.create_pool('pool_3', drop_3)

        res = TimeDistributed(Flatten())(pool_3)

        gru_1 = LipNext.create_bi_gru('gru_1', res)
        gru_2 = LipNext.create_bi_gru('gru_2', gru_1)

        dense_1 = Dense(self.output_size, kernel_initializer=LipNext.CONV_KERNEL_INIT, name='dense_1')(gru_2)

        y_pred = LipNext.create_actv('softmax', dense_1, activation='softmax')

        input_labels = LipNext.create_input('labels', shape=[self.max_string])
        input_length = LipNext.create_input('input_length', shape=[1], dtype='int64')
        label_length = LipNext.create_input('label_length', shape=[1], dtype='int64')

        loss_out = CTC('ctc', [y_pred, input_labels, input_length, label_length])

        self.model = Model(inputs=[input_layer, input_labels, input_length, label_length],
                           outputs=loss_out)

    def get_input_shape(self):
        if k.image_data_format() == 'channels_first':
            return self.image_channels, self.frame_rate, self.image_width, self.image_height
        else:
            return self.frame_rate, self.image_width, self.image_height, self.image_channels

    def summary(self):
        print(self.model.summary())

    def plot_model(self, file_name: str = 'model.png'):
        plot_model(lipnext.model, to_file=file_name, show_shapes=True)

    @staticmethod
    def create_input(name: str, shape, dtype: str = INPUT_TYPE) -> Input:
        return Input(shape=shape, dtype=dtype, name=name)

    @staticmethod
    def create_zero(name: str, input_layer, padding: tuple = ZERO_PADDING) -> ZeroPadding3D:
        return ZeroPadding3D(padding=padding, name=name)(input_layer)

    @staticmethod
    def create_conv(name: str, input_layer, filters: int, kernel_size: tuple = CONV_KERNEL_SIZE) -> Conv3D:
        return Conv3D(filters, kernel_size, strides=LipNext.CONV_STRIDES, kernel_initializer=LipNext.CONV_KERNEL_INIT,
                      name=name)(input_layer)

    @staticmethod
    def create_batc(name: str, input_layer) -> BatchNormalization:
        return BatchNormalization(name=name)(input_layer)

    @staticmethod
    def create_actv(name: str, input_layer, activation: str = CONV_ACTIVATION) -> Activation:
        return Activation(activation, name=name)(input_layer)

    @staticmethod
    def create_pool(name: str, input_layer) -> MaxPooling3D:
        return MaxPooling3D(pool_size=LipNext.POOL_SIZE, strides=LipNext.POOL_STRIDES, name=name)(input_layer)

    @staticmethod
    def create_drop(name: str, input_layer) -> SpatialDropout3D:
        return SpatialDropout3D(LipNext.DROPOUT_RATE, name=name)(input_layer)

    @staticmethod
    def create_bi_gru(name: str, input_layer, units: int = 256) -> Bidirectional:
        return Bidirectional(GRU(units, return_sequences=True, kernel_initializer=LipNext.GRU_KERNEL_INIT, name=name),
                             merge_mode='concat')(input_layer)


if __name__ == '__main__':
    lipnext = LipNext()

    lipnext.plot_model()
    lipnext.summary()
