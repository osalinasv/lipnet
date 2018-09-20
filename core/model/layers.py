from keras import backend as k
from keras.layers import Input
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.core import Activation, Dense, Flatten, Lambda, SpatialDropout3D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling3D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed


INPUT_TYPE = 'float32'

ZERO_PADDING = (1, 2, 2)

ACTIVATION_FN = 'relu'

CONV_KERNEL_INIT = 'he_normal'
CONV_KERNEL_SIZE = (3, 5, 5)
CONV_STRIDES = (1, 2, 2)

POOL_SIZE = (1, 2, 2)
POOL_STRIDES = (1, 2, 2)

DROPOUT_RATE = 0.5

GRU_ACTIVATION = None
GRU_UNITS = 256
GRU_KERNEL_INIT = 'Orthogonal'
GRU_MERGE_MODE  = 'concat'


def create_input_layer(name: str, shape, dtype: str = INPUT_TYPE) -> Input:
	return Input(shape=shape, dtype=dtype, name=name)


def create_zero_layer(name: str, input_layer, padding: tuple = ZERO_PADDING) -> ZeroPadding3D:
	return ZeroPadding3D(padding=padding, name=name)(input_layer)


def create_conv_layer(name: str, input_layer, filters: int, kernel_size: tuple = CONV_KERNEL_SIZE) -> Conv3D:
	return Conv3D(filters, kernel_size, strides=CONV_STRIDES, kernel_initializer=CONV_KERNEL_INIT, name=name)(input_layer)


def create_batc_layer(name: str, input_layer) -> BatchNormalization:
	return BatchNormalization(name=name)(input_layer)


def create_actv_layer(name: str, input_layer, activation: str = ACTIVATION_FN) -> Activation:
	return Activation(activation, name=name)(input_layer)


def create_pool_layer(name: str, input_layer) -> MaxPooling3D:
	return MaxPooling3D(pool_size=POOL_SIZE, strides=POOL_STRIDES, name=name)(input_layer)


def create_drop_layer(name: str, input_layer) -> SpatialDropout3D:
	return SpatialDropout3D(DROPOUT_RATE, name=name)(input_layer)


def create_bi_gru_layer(name: str, input_layer, units: int = GRU_UNITS, activation: str = GRU_ACTIVATION) -> Bidirectional:
	return Bidirectional(GRU(units, return_sequences=True, activation=activation, kernel_initializer=GRU_KERNEL_INIT, name=name), merge_mode='concat')(input_layer)


def create_timed_layer(input_layer) -> TimeDistributed:
	return TimeDistributed(Flatten())(input_layer)


def create_dense_layer(name: str, input_layer, output_size, kernel_initializer=CONV_KERNEL_INIT) -> Dense:
	return Dense(output_size, kernel_initializer=kernel_initializer, name=name)(input_layer)


def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	y_pred = y_pred[:, :, :]
	
	return k.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(name: str, args) -> Lambda:
	return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)


def create_ctc_layer(name: str, y_pred, input_labels, input_length, label_length) -> Lambda:
	return ctc(name, [y_pred, input_labels, input_length, label_length])
