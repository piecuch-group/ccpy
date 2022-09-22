"""
CCpy
A Python framework for coupled-cluster computations of molecular systems
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = "A Python framework for coupled-cluster computations of molecular systems".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = None

setup(
    # Self-descriptive entries which should always be present
    name='ccpy',
    author='Karthik Gururangan',
    author_email='gururang@msu.edu',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='BSD-3-Clause',

    packages=find_packages(),

    include_package_data=True,

    setup_requires=[] + pytest_runner,
)
