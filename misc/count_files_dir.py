import os
import sys

from colorama import Fore, Style


directory_path = os.path.realpath(sys.argv[1])
path_root, dirs, _ = next(os.walk(directory_path))

dir_list = []

for d in dirs:
	path, _, files = next(os.walk(os.path.join(path_root, d)))
	file_count = len(files)

	sizes = [os.path.getsize(os.path.join(path, file)) for file in files]
	average_size = (sum(sizes) / len(sizes)) / 1024

	dir_list.append((path, file_count, average_size))

total_average_size = sum([x[2] for x in dir_list]) / len(dir_list)
ta_small_p  = total_average_size * 0.05
ta_big_p    = total_average_size * 0.1

for p, fc, a in dir_list:
	average_size_int = int(round(a))
	average_size_str = '{} KB'.format(average_size_int) + Style.RESET_ALL

	diff = total_average_size - a

	if diff >= ta_small_p:
		average_size_str = Fore.YELLOW + average_size_str
	elif diff >= ta_small_p:
		average_size_str = Fore.RED + average_size_str

	print('{}\t => {} files, {} average size'.format(p, fc, average_size_str))
