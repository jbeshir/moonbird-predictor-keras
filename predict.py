import keras.models
from sklearn.externals import joblib
import predictor

model = keras.models.load_model("model.hdf5")
scaler = joblib.load("scaler.save")

print(predictor.predict(model, [0.01], scaler))
print(predictor.predict(model, [0.01, 0.01], scaler))
print(predictor.predict(model, [0.01, 0.01, 0.01], scaler))
print(predictor.predict(model, [0.01, 0.01, 0.01, 0.01], scaler))
print(predictor.predict(model, [0.99, 0.99], scaler))
print(predictor.predict(model, [0.5], scaler))
print(predictor.predict(model, [0.5, 0.5], scaler))
print(predictor.predict(model, [0.5, 0.5, 0.5], scaler))
print(predictor.predict(model, [0.99, 0.01], scaler))
print(predictor.predict(model, [0.01, 0.99], scaler))
