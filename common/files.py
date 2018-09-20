import fnmatch
import os


def is_dir(path: str) -> bool:
	return isinstance(path, str) and os.path.exists(path) and os.path.isdir(path)


def is_file(path: str) -> bool:
	return isinstance(path, str) and os.path.exists(path) and os.path.isfile(path)


def get_file_extension(path: str) -> str:
	return os.path.splitext(path)[1] if is_file(path) else ''


def get_file_name(path: str) -> str:
	return os.path.splitext(os.path.basename(path))[0] if is_file(path) else ''


def make_dir_if_not_exists(path: str):
	if not is_dir(path): os.makedirs(path)


def get_files_in_dir(path: str, pattern: str = '*'):
	for root, _, files in os.walk(path):
		for f in files:
			if fnmatch.fnmatch(f, pattern):
				yield os.path.realpath(os.path.join(root, f))


def get_immediate_subdirs(path: str) -> [str]:
	return [os.path.join(path, s) for s in next(os.walk(path))[1]] if is_dir(path) else []
