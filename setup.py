'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='moonbird_predictor_keras',
      version='1.1',
      packages=find_packages(),
      include_package_data=True,
      description='Moonbird Predictor keras model on Cloud ML Engine',
      author='John Beshir',
      author_email='john@beshir.org',
      license='MIT',
      install_requires=[
          'h5py==2.9.0',
          'joblib==0.13.2',
          'numpy==1.16.5',
		  'pandas==0.24.2',
		  'scikit-learn==0.21.3',
          'tensorflow==1.14.0'],
      zip_safe=False)