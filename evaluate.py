import keras.models
import numpy as np
from sklearn.externals import joblib
import predictor

data_path = "./data/"

# Hyperparameters
padvalue = -1

model = keras.models.load_model("model.hdf5")
scaler = joblib.load("scaler.save")

# Load CV data and evaluate
Xcv, Ycv, _ = predictor.loaddata(data_path + 'summarydata-cv.csv', data_path + "responsedata.csv", scaler, padvalue)
print(model.evaluate(Xcv, Ycv, batch_size=128))

Ycv_avg = predictor.avg_probability(Xcv)
print("Naive averaging MSE:", np.mean(np.square(Ycv_avg - Ycv), axis=0))
