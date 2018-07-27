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
		'dlib==19.15.0',
		'h5py==2.8.0',
		'imutils==0.4.6',
		'Keras==2.1.6',
		'matplotlib==2.2.2',
		'numpy==1.15.0',
		'opencv-python==3.4.1.15',
		'Pillow==5.1.0',
		'pymediainfo==2.3.0',
		'python-dotenv==0.8.2',
		'progress==1.4',
		'scikit-image==0.14.0',
		'scikit-video==1.1.10',
		'scipy==1.1.0',
		'tensorflow-gpu==1.9.0'
	]
)
