import keras.models
from sklearn.externals import joblib
import predictor

model = keras.models.load_model("model.hdf5")
scaler = joblib.load("scaler.save")

print(predictor.predict(model, [0.9, 0.1, 0.1], scaler))
