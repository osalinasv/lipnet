from itertools import chain, islice


def chunks(iterable, size: int = 1):
	iterator = iter(iterable)
	for first in iterator:
		yield chain([first], islice(iterator, size - 1))
