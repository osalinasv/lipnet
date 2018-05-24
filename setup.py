from setuptools import setup

setup(
    name='Lipnet',
    version='0.1',
    license='Apache License, Version 2.0',
    long_description=open('README.md').read(),
    url="http://www.github.com/omarsalinas16/lipnext",
    packages=['lipnext', 'preprocessing'],
    zip_safe=False,
    install_requires=[
        'dlib',
        'h5py',
        'Keras',
        'matplotlib',
        'numpy',
        'Pillow',
        'scipy',
        'scikit-image',
        'sk-video',
        'tensorflow-gpu'
    ]
)
