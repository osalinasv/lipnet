from setuptools import setup

setup(
    name='Lipnet',
    version='0.1',
    license='Apache License, Version 2.0',
    long_description=open('README.md').read(),
    url="http://www.github.com/omarsalinas16/lipnext",
    packages=['lipnext'],
    install_requires=[
        'numpy',
        'keras',
        'tensorflow',
        'tensorflow-gpu'
    ]
)
