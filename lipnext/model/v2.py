import lipnext.model.layers as layers
import env

from keras import backend as k
from keras.models import Model
from keras.optimizers import Adam


ADAM_LEARN_RATE = 0.0001
ADAM_F_MOMENTUM = 0.9
ADAM_S_MOMENTUM = 0.999
ADAM_STABILITY  = 1e-08


def create_model(frame_count: int, image_channels: int, image_height: int, image_width: int, max_string: int, output_size: int = env.OUTPUT_SIZE) -> Model:
	input_shape = get_input_shape(frame_count, image_channels, image_height, image_width)
	input_layer = layers.create_input_layer('input', input_shape)

	zero_1 = layers.create_zero_layer('zero_1', input_layer)
	conv_1 = layers.create_conv_layer('conv_1', zero_1, 32)
	batc_1 = layers.create_batc_layer('batc_1', conv_1)
	actv_1 = layers.create_actv_layer('actv_1', batc_1)
	drop_1 = layers.create_drop_layer('drop_1', actv_1)
	pool_1 = layers.create_pool_layer('pool_1', drop_1)

	zero_2 = layers.create_zero_layer('zero_2', pool_1)
	conv_2 = layers.create_conv_layer('conv_2', zero_2, 64)
	batc_2 = layers.create_batc_layer('batc_2', conv_2)
	actv_2 = layers.create_actv_layer('actv_2', batc_2)
	drop_2 = layers.create_drop_layer('drop_2', actv_2)
	pool_2 = layers.create_pool_layer('pool_2', drop_2)

	zero_3 = layers.create_zero_layer('zero_3', pool_2, padding=(1, 1, 1))
	conv_3 = layers.create_conv_layer('conv_3', zero_3, 96, kernel_size=(3, 3, 3))
	batc_3 = layers.create_batc_layer('batc_3', conv_3)
	actv_3 = layers.create_actv_layer('actv_3', batc_3)
	drop_3 = layers.create_drop_layer('drop_3', actv_3)
	pool_3 = layers.create_pool_layer('pool_3', drop_3)

	res = layers.create_timed_layer(pool_3)

	gru_1 = layers.create_bi_gru_layer('gru_1', res)
	gru_2 = layers.create_bi_gru_layer('gru_2', gru_1)

	dense_1 = layers.create_dense_layer('dense_1', gru_2, output_size)

	y_pred = layers.create_actv_layer('softmax', dense_1, activation='softmax')

	input_labels = layers.create_input_layer('labels', shape=[max_string])
	input_length = layers.create_input_layer('input_length', shape=[1], dtype='int64')
	label_length = layers.create_input_layer('label_length', shape=[1], dtype='int64')

	loss_out = layers.create_ctc_layer('ctc', y_pred, input_labels, input_length, label_length)

	return Model(inputs=[input_layer, input_labels, input_length, label_length], outputs=loss_out)


def compile_model(model: Model, optimizer = None):
	if optimizer == None:
		optimizer = Adam(lr=ADAM_LEARN_RATE, beta_1=ADAM_F_MOMENTUM, beta_2=ADAM_S_MOMENTUM, epsilon=ADAM_STABILITY)

	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)


def get_input_shape(frame_count: int, image_channels: int, image_height: int, image_width: int) -> tuple:
	if k.image_data_format() == 'channels_first':
		return image_channels, frame_count, image_width, image_height
	else:
		return frame_count, image_width, image_height, image_channels


if __name__ == '__main__':
	model = create_model(frame_count=75, image_channels=3, image_height=50, image_width=100, max_string=32)
	model.summary()

	compile_model(model)
