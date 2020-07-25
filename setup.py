from setuptools import setup


setup(
    name='psf-weather-station',
    version='0.1',
    description='generate realistic atmospheric input parameters for point-spread function simulations',
    url='http://github.com/cahebert/psf-weather-station',
    author='claire',
    author_email='chebert@stanford.edu',
    license='MIT',
    packages=['pws'],
    python_requires='>=3.5',
    zip_safe=False
)
