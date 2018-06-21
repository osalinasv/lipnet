import os
import fnmatch

def make_dir(path: str):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)


def find_files(path: str, pattern: str):
    for root, _, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.realpath(os.path.join(root, basename))
                yield filename

def walklevel(some_dir, level=1):
	some_dir = some_dir.rstrip(os.path.sep)

	assert os.path.isdir(some_dir)

	num_sep = some_dir.count(os.path.sep)

	for root, dirs, files in os.walk(some_dir):
		yield root, dirs, files

		num_sep_this = root.count(os.path.sep)

		if num_sep + level <= num_sep_this:
			del dirs[:]

def read_subfolders(path):
	subfolders = []

	for subdir, _, _ in walklevel(path):
		if subdir == path:
			continue

		subfolders.append(subdir)

	return subfolders
