import argparse
import fnmatch
import os
import random
import shutil

from colorama import init, Fore


init(autoreset=True)


# python misc\copy_files.py -i data\dataset_removed\ -o data\dataset_eval -a 64

def get_npy_files(path: str) -> [str]:
	for _, _, files in os.walk(path):
		for f in files:
			if fnmatch.fnmatch(f, '*.npy'):
				yield os.path.realpath(os.path.join(path, f))


def copy_files(inp: str, out: str, amount: int) -> (str, [str]):
	for root, _, _ in os.walk(inp):
		if root == inp: continue

		file_paths = [f for f in get_npy_files(root)]
		max_amount = min(amount, len(file_paths))

		random_takes = random.sample(file_paths, max_amount)

		print('TAKEN [{}] FROM: {}'.format(len(random_takes), root))

		yield root, random_takes


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument('-i', '--input-path', required=True,
		help='Path to the source dataset')

	ap.add_argument('-o', '--output-path', required=True,
		help='Path to the output dataset')

	ap.add_argument('-a', '--amount', required=True,
		help='Amount of files to copy per subdirectory', type=int)

	args = vars(ap.parse_args())

	input_path  = os.path.realpath(args['input_path'])
	output_path = os.path.realpath(args['output_path'])
	amount = max(1, args['amount'])

	print('Searching for files at: {}\n'.format(input_path))

	print('Will copy files to: {}'.format(output_path))
	print('Copying {} files per subdirectory\n'.format(amount))

	if not os.path.exists(input_path) or not os.path.isdir(input_path):
		print(Fore.RED + '\nERROR: Input path is not a directory')
		return

	if os.path.exists(output_path) and os.path.isdir(output_path):
		shutil.rmtree(output_path)

	os.makedirs(output_path)

	for _, files in copy_files(input_path, output_path, amount):
		for f in files:
			new_f = f.replace(input_path, output_path)
			new_d = os.path.dirname(new_f)

			if not os.path.exists(new_d) or not os.path.isdir(new_d):
				os.makedirs(new_d)

			shutil.copyfile(f, new_f)


if __name__ == '__main__':
	main()
