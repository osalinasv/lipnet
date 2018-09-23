import numpy as np

from core.utils.labels import text_to_labels
from typing import NamedTuple


__SILENCE_TOKENS = ['sp', 'sil']


class Align(NamedTuple):
	sentence: str
	labels:   np.ndarray
	length:   int


def align_from_file(path: str, max_string: int) -> Align:
	with open(path, 'r') as f:
		lines = f.readlines()

	align = [(int(y[0]) / 1000, int(y[1]) / 1000, y[2]) for y in [x.strip().split(' ') for x in lines]]
	align = __strip_from_align(align, __SILENCE_TOKENS)

	sentence = __get_align_sentence(align, __SILENCE_TOKENS)
	labels   = __get_sentence_labels(sentence)
	padded_labels = __get_padded_label(labels, max_string)

	return Align(sentence, padded_labels, len(labels))


def __strip_from_align(align: list, items: list) -> list:
	return [sub for sub in align if sub[2] not in items]


def __get_align_sentence(align: list, items: list) -> str:
	return ' '.join([y[-1] for y in align if y[-1] not in items])


def __get_sentence_labels(sentence: str) -> list:
	return text_to_labels(sentence)


def __get_padded_label(labels: list, max_string: int) -> np.ndarray:
	return np.array(labels + ([-1.0] * (max_string - len(labels))))
