'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='moonbird_predictor_keras',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Moonbird Predictor keras model on Cloud ML Engine',
      author='John Beshir',
      author_email='john@beshir.org',
      license='MIT',
      install_requires=[
          'h5py',
          'keras',
          'numpy',
		  'pandas',
		  'scikit-learn',
          'tensorflow'],
      zip_safe=False)