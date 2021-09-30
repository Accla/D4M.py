from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='D4M',
    version='1.2.2',
    description='Python implementation of D4M',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Accla/D4M.py',
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    include_package_data=True,
    zip_safe=False,
    packages=find_packages(),
    install_requires=['scipy', 'numpy', 'py4j', 'matplotlib'],
)
