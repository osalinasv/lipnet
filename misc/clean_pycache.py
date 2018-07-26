import os
import sys
import shutil


target_path = os.path.realpath(sys.argv[1])
pycaches = []

for root, dirs, _ in os.walk(target_path):
	for d in dirs:
		if d == '__pycache__':
			pycaches.append(os.path.realpath(os.path.join(root, d)))

for d in pycaches:
	print('Deleting {}'.format(d))
	shutil.rmtree(d)
