from setuptools import setup

setup(
	name             = 'lipnext',
	version          = '0.1',
	license          = 'Apache License, Version 2.0',
	long_description = open('README.md').read(),
	url              = "http://www.github.com/omarsalinas16/lipnext",
	packages         = ['common', 'lipnext', 'preprocessing'],
	zip_safe         = False,
	install_requires = [
		'colorama==0.3.9',
		'dlib==19.15.0',
		'editdistance==0.4',
		'imutils==0.5.1',
		'Keras==2.2.2',
		'Keras-Applications==1.0.4',
		'Keras-Preprocessing==1.0.2',
		'matplotlib==2.2.3',
		'numpy==1.14.5',
		'opencv-python==3.4.1.15',
		'python-dotenv==0.9.1',
		'progress==1.4',
		'sk-video==1.1.10',
		'tensorflow-gpu==1.10.0'
	]
)
