import io
import os
import argparse
import numpy as np
import h5py
from tensorflow.python.lib.io import file_io
from sklearn.externals import joblib

import trainer.export_to_tensorflow as export_to_tensorflow
import trainer.predictor as predictor


def train_model(train_file='./data/', job_dir='./job/', glove_file=None, num_epochs=100, **args):

    embeddings_index = None
    if glove_file is not None:
        embeddings_index = predictor.loadembeddings(glove_file)

    # Build model
    model = predictor.buildmodel_defaults()
    print(model.summary())

    # Load training data and train
    Xtrain, Ytrain, scaler = predictor.loaddata(train_file + 'summarydata-train.csv', train_file + "responsedata.csv", embeddings_index, True)
    model.fit(Xtrain, Ytrain, epochs=num_epochs, batch_size=128)

    # Load CV data and evaluate
    Xcv, Ycv, _ = predictor.loaddata(train_file + 'summarydata-cv.csv', train_file + "responsedata.csv", embeddings_index, scaler)
    print(model.evaluate(Xcv, Ycv, batch_size=128))

    Xcv_avg, _, _ = predictor.loaddata(train_file + 'summarydata-cv.csv', train_file + "responsedata.csv", embeddings_index, False)
    Ycv_avg = predictor.avg_probability(Xcv_avg)
    print("Naive averaging MSE:", np.mean(np.square(Ycv_avg - Ycv), axis=0))

    # Because H5PY fails if we try to pass it a gcsio object directly.
    modelBuf = io.BytesIO()
    model.save(h5py.File(modelBuf))
    with write_local_or_gcs(job_dir + "model/model.hdf5") as f:
        f.write(modelBuf.read())

    joblib.dump(scaler, write_local_or_gcs(job_dir + "model/scaler.save"))

    # Export to TensorFlow Saved Model suitable for serving.
    export_to_tensorflow.to_savedmodel(model, scaler, job_dir + "saved_model/")


def write_local_or_gcs(path):
    if path.startswith("gs://"):
        return file_io.FileIO(path, 'wb')
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, 'wb')


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
      '--glove-file',
      help='Path to the glove.6B.50d.txt file containing our word embeddings',
      default=None)
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
