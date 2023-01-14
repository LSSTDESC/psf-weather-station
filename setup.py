from setuptools import setup
import os

# stole this form ImSim:
def all_files_from(dir, ext=''):
    """Quick function to get all files from directory and all subdirectories
    """
    files = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(ext) and not filename.startswith('.'):
                files.append(os.path.join(root, filename))
    return files

data_files = all_files_from('data')

setup(
    name='psfws',
    version='0.1',
    description='generate realistic wind and turbulence parameters for atmospheric point-spread function simulations',
    url='https://github.com/LSSTDESC/psf-weather-station',
    author='Claire-Alice Hebert',
    author_email='chebert@stanford.edu',
    license='MIT',
    packages=['psfws'],
    package_data={'psfws': data_files},
    python_requires='>=3.5'
)
