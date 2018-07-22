import os
import fnmatch


def is_dir(path: str) -> bool:
	return isinstance(path, str) and os.path.exists(path) and os.path.isdir(path)


def is_file(path: str) -> bool:
	return isinstance(path, str) and os.path.exists(path) and os.path.isfile(path)


def get_file_extension(path: str) -> str:
	return os.path.splitext(path)[1]


def get_file_name(path: str) -> str:
	return os.path.splitext(os.path.basename(path))[0]


def make_dir_if_not_exists(path: str):
	if not is_dir(path): os.makedirs(path)


def get_files_in_dir(path: str, pattern: str):
	for root, _, files in os.walk(path):
		for basename in files:
			if fnmatch.fnmatch(basename, pattern):
				yield os.path.realpath(os.path.join(root, basename))


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
