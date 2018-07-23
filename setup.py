from setuptools import setup

setup(
	name             = 'lipnext',
	version          = '0.1',
	license          = 'Apache License, Version 2.0',
	long_description = open('README.md').read(),
	url              = "http://www.github.com/omarsalinas16/lipnext",
	packages         = ['lipnext', 'preprocessing', 'evaluation', 'training'],
	zip_safe         = False,
	install_requires = [
		'colorama==0.3.9',
		'dlib==19.8.1',
		'h5py==2.7.1',
		'imutils==0.4.6',
		'Keras==2.1.6',
		'matplotlib==2.2.2',
		'numpy==1.14.3',
		'opencv-python==3.4.1.15',
		'Pillow==5.1.0',
		'progress==1.3',
		'scikit-image==0.13.1',
		'scikit-video==1.1.10',
		'scipy==1.1.0',
		'tensorflow-gpu==1.8.0'
	]
)
