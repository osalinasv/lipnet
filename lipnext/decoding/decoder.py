import numpy as np

from keras import backend as k


class Decoder(object):

	def __init__(self, greedy: bool = True, beam_width: int = 100, top_paths: int = 1, postprocessors: list = []):
		self.greedy         = greedy
		self.beam_width     = beam_width
		self.top_paths      = top_paths
		self.postprocessors = postprocessors


	def decode(self, y_pred: np.ndarray, input_length: int) -> list:
		decoded = self._decode(y_pred, input_length, self.greedy, self.beam_width, self.top_paths)

		preprocessed = []

		for output in decoded:
			for postprocessor in self.postprocessors:
					output = postprocessor(output)
			preprocessed.append(output)

		return preprocessed


	def _decode(self, y_pred: np.ndarray, input_length: int, greedy: bool, beam_width: int, top_paths: int) -> list:
		paths, _ = self._keras_decode(y_pred, input_length, greedy, beam_width, top_paths)

		result = paths[0]
		return result


	def _keras_decode(self, y_pred: np.ndarray, input_length: int, greedy: bool, beam_width: int, top_paths: int) -> list:
		decoded = k.ctc_decode(y_pred=y_pred, input_length=input_length, greedy=greedy, beam_width=beam_width, top_paths=top_paths)

		paths    = [path.eval(session=k.get_session()) for path in decoded[0]]
		logprobs = decoded[1].eval(session=k.get_session())

		return paths, logprobs
