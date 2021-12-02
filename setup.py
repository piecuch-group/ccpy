from setuptools import setup, find_packages
#import versioneer

setup(
    name='CCpy',
    version='0.1',
    description='A Python framework for coupled-cluster computations of molecular systems',
    author='Karthik Gururangan',
    author_email='gururang@msu.edu',
    packages=find_packages(),
    long_description=open('README.md').read(),
    install_requires=[
            'numpy',
            'mkl',
            'cclib',
            'scipy',
    ])

