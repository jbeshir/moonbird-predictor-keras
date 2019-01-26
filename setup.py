'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='moonbird-predictor-keras',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Moonbird Predictor keras model on Cloud ML Engine',
      author='John Beshir',
      author_email='john@beshir.org',
      license='MIT',
      install_requires=[
          'keras',
		  'pandas',
		  'scikit-learn'],
      zip_safe=False)