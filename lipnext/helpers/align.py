import numpy as np


class Align(object):

	def __init__(self, absolute_max_string_len: int):
		self.absolute_max_string_len = absolute_max_string_len


	def from_file(self, path: str):
		with open(path, 'r') as f:
			lines = f.readlines()

		align = [(int(y[0]) / 1000, int(y[1]) / 1000, y[2]) for y in [x.strip().split(' ') for x in lines]]
		self.build(align)

		return self


	def build(self, align: list):
		self.align        = self.strip(align, ['sp','sil'])
		self.sentence     = self.get_sentence(align)
		self.label        = self.get_label(self.sentence)
		self.padded_label = self.get_padded_label(self.label)


	def strip(self, align, items):
		return [sub for sub in align if sub[2] not in items]


	def get_sentence(self, align):
		return ' '.join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])


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
		padding = np.ones((self.absolute_max_string_len - len(label))) * -1
		return np.concatenate((np.array(label), padding), axis=0)


	@property
	def label_length(self):
		return len(self.label)
