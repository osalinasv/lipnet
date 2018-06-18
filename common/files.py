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
