import os
import sys


directory_path = os.path.realpath(sys.argv[1])
path_root, dirs, _ = next(os.walk(directory_path))

for d in dirs:
	path, _, files = next(os.walk(os.path.join(path_root, d)))
	file_count = len(files)

	sizes = [os.path.getsize(os.path.join(path, file)) for file in files]
	average_size = (sum(sizes) / len(sizes)) / 1024

	print('{}\t -> {} files, {:.0f} KB average size'.format(path, file_count, average_size))
