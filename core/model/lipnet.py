from keras import backend as k
from keras.models import Model
from keras.optimizers import Adam

import core.model.layers as layers
import env


ADAM_LEARN_RATE = 0.0001
ADAM_F_MOMENTUM = 0.9
ADAM_S_MOMENTUM = 0.999
ADAM_STABILITY  = 1e-08


class LipNet(object):

	def __init__(self, frame_count: int, image_channels: int, image_height: int, image_width: int, max_string: int, output_size: int = env.OUTPUT_SIZE):
		input_shape = self.get_input_shape(frame_count, image_channels, image_height, image_width)
		self.input_layer = layers.create_input_layer('input', input_shape)

		self.zero_1 = layers.create_zero_layer('zero_1', self.input_layer)
		self.conv_1 = layers.create_conv_layer('conv_1', self.zero_1, 32)
		self.batc_1 = layers.create_batc_layer('batc_1', self.conv_1)
		self.actv_1 = layers.create_actv_layer('actv_1', self.batc_1)
		self.pool_1 = layers.create_pool_layer('pool_1', self.actv_1)
		self.drop_1 = layers.create_drop_layer('drop_1', self.pool_1)

		self.zero_2 = layers.create_zero_layer('zero_2', self.drop_1)
		self.conv_2 = layers.create_conv_layer('conv_2', self.zero_2, 64)
		self.batc_2 = layers.create_batc_layer('batc_2', self.conv_2)
		self.actv_2 = layers.create_actv_layer('actv_2', self.batc_2)
		self.pool_2 = layers.create_pool_layer('pool_2', self.actv_2)
		self.drop_2 = layers.create_drop_layer('drop_2', self.pool_2)

		self.zero_3 = layers.create_zero_layer('zero_3', self.drop_2, padding=(1, 1, 1))
		self.conv_3 = layers.create_conv_layer('conv_3', self.zero_3, 96, kernel_size=(3, 3, 3))
		self.batc_3 = layers.create_batc_layer('batc_3', self.conv_3)
		self.actv_3 = layers.create_actv_layer('actv_3', self.batc_3)
		self.pool_3 = layers.create_pool_layer('pool_3', self.actv_3)
		self.drop_3 = layers.create_drop_layer('drop_3', self.pool_3)

		self.res = layers.create_timed_layer(self.drop_3)

		self.gru_1 = layers.create_bi_gru_layer('gru_1', self.res)
		self.gru_1_actv = layers.create_actv_layer('gru_1_actv', self.gru_1)
		self.gru_2 = layers.create_bi_gru_layer('gru_2', self.gru_1_actv)
		self.gru_2_actv = layers.create_actv_layer('gru_2_actv', self.gru_2)

		self.dense_1 = layers.create_dense_layer('dense_1', self.gru_2_actv, output_size)
		self.y_pred  = layers.create_actv_layer('softmax', self.dense_1, activation='softmax')

		self.input_labels = layers.create_input_layer('labels', shape=[max_string])
		self.input_length = layers.create_input_layer('input_length', shape=[1], dtype='int64')
		self.label_length = layers.create_input_layer('label_length', shape=[1], dtype='int64')

		self.loss_out = layers.create_ctc_layer('ctc', self.y_pred, self.input_labels, self.input_length, self.label_length)

		self.model = Model(inputs=[self.input_layer, self.input_labels, self.input_length, self.label_length], outputs=self.loss_out)


	def compile_model(self, optimizer=None):
		if optimizer is None:
			optimizer = Adam(lr=ADAM_LEARN_RATE, beta_1=ADAM_F_MOMENTUM, beta_2=ADAM_S_MOMENTUM, epsilon=ADAM_STABILITY)

		self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
		return self


	def load_weights(self, path: str):
		self.model.load_weights(path)
		return self


	@staticmethod
	def get_input_shape(frame_count: int, image_channels: int, image_height: int, image_width: int) -> tuple:
		if k.image_data_format() == 'channels_first':
			return image_channels, frame_count, image_width, image_height
		else:
			return frame_count, image_width, image_height, image_channels


	def predict(self, input_batch):
		return self.capture_softmax_output([input_batch, 0])[0]


	@property
	def capture_softmax_output(self):
		return k.function([self.input_layer, k.learning_phase()], [self.y_pred])
