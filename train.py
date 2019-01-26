import numpy as np
import argparse
import predictor
from sklearn.externals import joblib

import export_to_tensorflow

def train_model(train_file='./data/', model_save_path='./model/',
                job_dir='./tf-export/', **args):

    embeddings_index = predictor.loadembeddings('glove.6B.50d.txt')

    # Build model
    model = predictor.buildmodel_defaults()
    print(model.summary())

    # Load training data and train
    Xtrain, Ytrain, scaler = predictor.loaddata(train_file + 'summarydata-train.csv', train_file + "responsedata.csv", embeddings_index, True)
    model.fit(Xtrain, Ytrain, epochs=100, batch_size=128)

    # Load CV data and evaluate
    Xcv, Ycv, _ = predictor.loaddata(train_file + 'summarydata-cv.csv', train_file + "responsedata.csv", embeddings_index, scaler)
    print(model.evaluate(Xcv, Ycv, batch_size=128))

    Xcv_avg, _, _ = predictor.loaddata(train_file + 'summarydata-cv.csv', train_file + "responsedata.csv", embeddings_index, False)
    Ycv_avg = predictor.avg_probability(Xcv_avg)
    print("Naive averaging MSE:", np.mean(np.square(Ycv_avg - Ycv), axis=0))

    # Save
    model.save(model_save_path + "model.hdf5")
    joblib.dump(scaler, model_save_path + "scaler.save")

    # Export to TensorFlow Saved Model suitable for serving.
    export_to_tensorflow.to_savedmodel(model, scaler, job_dir)


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file',
      help='Cloud Storage bucket or local path to training data',
      default='./data/')
    parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files',
      default='./tf-export/')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
