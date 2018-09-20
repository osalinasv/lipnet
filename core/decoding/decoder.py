import numpy as np

from keras import backend as k


class Decoder(object):

	def __init__(self, greedy: bool = True, beam_width: int = 200, top_paths: int = 1, postprocessors=None):
		self.greedy         = greedy
		self.beam_width     = beam_width
		self.top_paths      = top_paths
		self.postprocessors = postprocessors if postprocessors is not None else []


	def decode(self, y_pred: np.ndarray, input_lengths: np.ndarray) -> list:
		decoded = self.__decode(y_pred, input_lengths, self.greedy, self.beam_width, self.top_paths)

		postprocessed = []

		for d in decoded:
			for f in self.postprocessors: d = f(d)
			postprocessed.append(d)

		return postprocessed


	def __decode(self, y_pred: np.ndarray, input_lengths: np.ndarray, greedy: bool, beam_width: int, top_paths: int) -> list:
		return self.__keras_decode(y_pred, input_lengths, greedy, beam_width, top_paths)[0]


	@staticmethod
	def __keras_decode(y_pred: np.ndarray, input_lengths: np.ndarray, greedy: bool, beam_width: int, top_paths: int) -> list:
		decoded = k.ctc_decode(y_pred=y_pred, input_length=input_lengths, greedy=greedy, beam_width=beam_width, top_paths=top_paths)

		return [path.eval(session=k.get_session()) for path in decoded[0]]
