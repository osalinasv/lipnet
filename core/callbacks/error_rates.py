import csv
import os

import editdistance
import numpy as np
from keras.callbacks import Callback
from keras.utils import Sequence

from common.files import make_dir_if_not_exists
from core.decoding.decoder import Decoder
from core.model.lipnet import LipNet
from core.utils.wer import wer_sentence


class ErrorRates(Callback):

	def __init__(self, output_path: str, lipnet: LipNet, val_generator: Sequence, decoder: Decoder, samples: int = 256):
		super().__init__()

		self.output_path = output_path
		self.lipnet     = lipnet
		self.generator   = val_generator.__getitem__
		self.decoder     = decoder
		self.samples     = samples


	def get_sample_batch(self) -> list:
		sample_batch  = []
		generator_idx = 0
		samples_left  = self.samples

		while samples_left > 0:
			batch = self.generator(generator_idx)[0]
			batch_input = batch['input']

			samples_to_take = min(len(batch_input), samples_left)

			if samples_to_take <= 0:
				break

			y_pred       = self.lipnet.predict(batch_input[0:samples_to_take])
			input_length = batch['input_length'][0:samples_to_take]

			decoded = self.decoder.decode(y_pred, input_length)

			for i in range(0, samples_to_take):
				sample_batch.append((decoded[i], batch['sentences'][i]))

			samples_left  -= samples_to_take
			generator_idx += 1

		return sample_batch


	@staticmethod
	def calculate_mean_generic(data: [tuple], mean_length: int, evaluator) -> (float, float):
		values = [float(evaluator(x[0], x[1])) for x in data]

		total = 0
		total_norm = 0

		for v in values:
			total += v
			total_norm += v / mean_length

		length = len(data)
		return total / length, total_norm / length


	def calculate_wer(self, data: [tuple]) -> (float, float):
		mean_length = int(np.mean([len(d[1].split()) for d in data]))
		return self.calculate_mean_generic(data, mean_length, wer_sentence)


	def calculate_cer(self, data: [tuple]) -> (float, float):
		mean_length = int(np.mean([len(d[1]) for d in data]))
		return self.calculate_mean_generic(data, mean_length, editdistance.eval)


	def calculate_statistics(self) -> dict:
		sample_batch = self.get_sample_batch()

		wer, wer_norm = self.calculate_wer(sample_batch)
		cer, cer_norm = self.calculate_cer(sample_batch)

		return {
			'samples':  len(sample_batch),
			'wer':      wer,
			'wer_norm': wer_norm,
			'cer':      cer,
			'cer_norm': cer_norm
		}


	def on_train_begin(self, logs=None):
		output_dir = os.path.dirname(self.output_path)
		make_dir_if_not_exists(output_dir)

		with open(self.output_path, 'w') as f:
			writer = csv.writer(f)
			writer.writerow(['epoch', 'samples', 'wer', 'wer_norm', 'cer', 'cer_norm'])


	def on_epoch_end(self, epoch: int, logs=None):
		print('Epoch {:05d}: Calculating error rates...'.format(epoch + 1), end='')

		statistics = self.calculate_statistics()

		print('\rEpoch {:05d}: ({} samples) [WER {:.3f} - {:.3f}]\t[CER {:.3f} - {:.3f}]\n'.format(epoch + 1, statistics['samples'], statistics['wer'], statistics['wer_norm'], statistics['cer'], statistics['cer_norm']))

		with open(self.output_path, 'a') as f:
			writer = csv.writer(f)
			writer.writerow([
				epoch,
				statistics['samples'],
				statistics['wer'],
				statistics['wer_norm'],
				statistics['cer'],
				statistics['cer_norm']
			])
