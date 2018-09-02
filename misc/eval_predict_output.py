import argparse
import csv
import editdistance
import numpy as np
import os

from colorama import init, Fore
from common.files import is_file, get_file_extension
from core.utils.wer import wer_sentence
from functools import reduce


init(autoreset=True)


# set PYTHONPATH=%PYTHONPATH%;./
# python misc\eval_predict_output.py -i output.csv -o predict_stats.csv -a data/aligns.csv

def get_csv_data(path: str) -> [dict]:
	with open(path, mode='r') as f:
		reader = csv.DictReader(f, delimiter=',')
		data = [r for r in reader]

	return data


def get_aligns_data(path: str) -> {str: str}:
	data = get_csv_data(path)
	data = dict(map(lambda x: (x['Video'], x['Sentence']), data))

	return data


def get_input_data(path: str) -> {str: str}:
	data = get_csv_data(path)

	file_to_id = lambda x: os.path.splitext(os.path.basename(x))[0]
	data = dict(map(lambda x: (file_to_id(x['file']), x['prediction']), data))

	return data


def calculate_mean_generic(data: [tuple], mean_length: int, evaluator) -> (float, float):
	values = list(map(lambda x: float(evaluator(x[0], x[1])), data))

	total = reduce(lambda x, y: x + y, values)
	total_norm = reduce(lambda x, y: x + (y / mean_length), values)

	length = len(data)
	return total / length, total_norm / length


def calculate_wer(data: [tuple]) ->  (float, float):
	mean_length = np.mean([len(d[1].split()) for d in data])
	return calculate_mean_generic(data, mean_length, wer_sentence)


def calculate_cer(data: [tuple]) ->  (float, float):
	mean_length = np.mean([len(d[1]) for d in data])
	return calculate_mean_generic(data, mean_length, editdistance.eval)


def generate_output_dataset(input_data: {str: str}, aligns_data: {str: str}) -> [tuple]:
	for i in input_data:
		r = aligns_data[i]
		h = input_data[i]

		wer = wer_sentence(r, h)
		cer = editdistance.eval(r, h)

		print('({})  {} : {}   WER={}    CER={}'.format(i, h.ljust(30, ' '), r.ljust(30, ' '), wer, cer))

		yield i, r, h, wer, cer


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument('-i', '--input-path', required=True,
		help='Path to the output CSV file')

	ap.add_argument('-o', '--output-path', required=True,
		help='Path to where the output will be saved')

	ap.add_argument('-a', '--aligns-path', required=True,
		help='Path to the aligns CSV file')

	args = vars(ap.parse_args())

	input_path = os.path.realpath(args['input_path'])
	output_path = os.path.realpath(args['output_path'])
	aligns_path = os.path.realpath(args['aligns_path'])

	if not is_file(input_path) or get_file_extension(input_path) != '.csv':
		print(Fore.RED + '\nERROR: Input path is not a CSV file')
		return

	if not is_file(aligns_path) or get_file_extension(aligns_path) != '.csv':
		print(Fore.RED + '\nERROR: Aligns path is not a CSV file')
		return

	input_data  = get_input_data(input_path)
	aligns_data = get_aligns_data(aligns_path)

	dataset = [x for x in generate_output_dataset(input_data, aligns_data)]

	with open(output_path, 'w') as f:
		writer = csv.writer(f)

		writer.writerow(['id', 'refference', 'hypothesis', 'wer', 'cer'])
		for x in dataset: writer.writerow([*x])


	sample_batch = [(x[2], x[1]) for x in dataset]

	wer, wer_norm = calculate_wer(sample_batch)
	cer, cer_norm = calculate_cer(sample_batch)

	print('\n\n[WER {:.3f} - {:.3f}]\t[CER {:.3f} - {:.3f}]'.format(wer, wer_norm, cer, cer_norm))


if __name__ == '__main__':
	main()