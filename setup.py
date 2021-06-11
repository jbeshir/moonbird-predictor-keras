'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='moonbird_predictor_keras',
      version='1.2',
      packages=find_packages(),
      include_package_data=True,
      description='Moonbird Predictor keras model on Cloud ML Engine',
      author='John Beshir',
      author_email='john@beshir.org',
      license='MIT',
      install_requires=[
          'h5py==2.10.0',
          'joblib==0.13.2',
          'numpy==1.19.4',
		      'pandas==1.2.3',
		      'scikit-learn==0.21.3',
          'tensorflow>=2.4.2'],
      zip_safe=False)