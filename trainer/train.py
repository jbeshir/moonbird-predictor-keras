import argparse
import os
import shutil
import tempfile
import keras
import numpy as np
import h5py
from tensorflow.python.lib.io import file_io
from sklearn.externals import joblib

import trainer.export_to_tensorflow as export_to_tensorflow
import trainer.predictor as predictor


def train_model(train_file='./data/', job_dir='./job/', prev_model_dir=None, glove_file=None, num_epochs=100, train_online_test=False, **args):

    embeddings_index = None
    if glove_file is not None:
        embeddings_index = predictor.loadembeddings(glove_file)

    if prev_model_dir is None:
        # Build model
        model = predictor.buildmodel_defaults()
        scaler = True
    else:
        # Load previous model iteration
        model = keras.models.load_model(h5py.File(open_local_or_gcs(prev_model_dir + "model.hdf5", 'rb'), mode='r'))
        scaler = joblib.load(open_local_or_gcs(prev_model_dir + "scaler.save", 'rb'))
    print(model.summary())

    # Load training data and train
    Xtrain, Ytrain, scaler = predictor.loaddata(train_file + 'summarydata-train.csv', train_file + "responsedata.csv", embeddings_index, scaler)
    XOnlineTrain, YOnlineTrain = None, None
    if train_online_test:
        splitPos = int(Xtrain.shape[0] * 0.995)
        Xtrain, XOnlineTrain = np.array_split(Xtrain, [splitPos])
        Ytrain, YOnlineTrain = np.array_split(Ytrain, [splitPos])
    model.fit(Xtrain, Ytrain, epochs=num_epochs, batch_size=128)
    if train_online_test:
        model.fit(XOnlineTrain, YOnlineTrain, epochs=1, batch_size=128)

    # Load CV data and evaluate
    if train_online_test:
        Xcv, Ycv, _ = predictor.loaddata(train_file + 'summarydata-cv.csv', train_file + "responsedata.csv", embeddings_index, scaler)
        print(model.evaluate(Xcv, Ycv, batch_size=128))

    Xcv_avg, _, _ = predictor.loaddata(train_file + 'summarydata-cv.csv', train_file + "responsedata.csv", embeddings_index, False)
    Ycv_avg = predictor.avg_probability(Xcv_avg)
    print("Naive averaging MSE:", np.mean(np.square(Ycv_avg - Ycv), axis=0))

    # H5PY produced corrupt files when passed a file-like object, at time/platform of development.
    # It also insists on being able to read, which breaks GCS-backed file-like objects.
    tempDir = tempfile.mkdtemp()
    try:
        tempFile = tempDir + os.path.sep + "model.hdf5"
        model.save(tempFile)

        with open_local_or_gcs(job_dir + "model/model.hdf5", 'wb') as f:
            with open(tempFile, 'rb') as tempFd:
                f.write(tempFd.read())
    finally:
        shutil.rmtree(tempDir)

    joblib.dump(scaler, open_local_or_gcs(job_dir + "model/scaler.save", 'wb'))

    # Export to TensorFlow Saved Model suitable for serving.
    export_to_tensorflow.to_savedmodel(model, scaler, job_dir + "saved_model/")


def open_local_or_gcs(path, mode):
    if path.startswith("gs://"):
        return file_io.FileIO(path, mode)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--job-dir',
      help='Cloud storage path to export the model and store temp files',
      default='./job/')
    parser.add_argument(
      '--train-file',
      help='Cloud Storage path or local path to training data',
      default='./data/')
    parser.add_argument(
      '--num-epochs',
      help='Number of epochs to run for',
      default=100,
      type=int)
    parser.add_argument(
      '--prev-model-dir',
      help='An existing model to use as a starting point for training. Used for incremental training.',
      default=None)
    parser.add_argument(
      '--glove-file',
      help='Path to the glove.6B.50d.txt file containing our word embeddings',
      default=None)
    parser.add_argument(
      '--train-online-test',
      help='Tests online training parameters',
	  action='store_true')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
