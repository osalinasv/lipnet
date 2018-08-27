import editdistance
import numpy as np

from functools import reduce
from lipnext.utils.wer import wer_sentence


def calculate_mean_generic(data: [tuple], mean_length: int, evaluator) -> (float, float):
	values = list(map(lambda x: float(evaluator(x[1], x[0])), data))

	print('values: {}'.format(values))

	total = reduce(lambda x, y: x + y, values)
	print('total: {}'.format(total))

	total_norm = reduce(lambda x, y: x + (y / mean_length), values)
	print('total_norm: {}'.format(total_norm))

	length = len(data)
	return total / length, total_norm / length


def calculate_wer(data: [tuple]) ->  (float, float):
	mean_length = np.mean([len(d[1].split()) for d in data])
	return calculate_mean_generic(data, mean_length, wer_sentence)


def calculate_cer(data: [tuple]) ->  (float, float):
	mean_length = np.mean([len(d[1]) for d in data])
	return calculate_mean_generic(data, mean_length, editdistance.eval)


data = [
	('bin blue at o two now', 'bin blue at f two now'),
	('bin white in y five soon', 'bin white in m five soon'),
	('lay green by e zero please', 'lay green by g zero please'),
	('place blue by m eight please', 'place blue by v eight please'),
	('place white at g zero now', 'place red in p zero now')
]

r = calculate_wer(data)
print('\nwer: {:.4f}\twer_norm: {:.4f}\twer_per: {:.4f}'.format(r[0], r[1], r[1] * 100.0))

print('\n')

r = calculate_cer(data)
print('\ncer: {:.4f}\tcer_norm: {:.4f}\tcer_per: {:.4f}'.format(r[0], r[1], r[1] * 100.0))


ref = 'This great machine can recognize speech'
hyp = 'This great machine can wreck a nice speech'

wer = wer_sentence(ref, hyp)
wer_norm = wer / len(ref.split())
wer_per = wer_norm * 100.0

print('\n')
print('wer: {:.4f}\twer_norm: {:.4f}\twer_per: {:.4f}'.format(wer, wer_norm, wer_per))
