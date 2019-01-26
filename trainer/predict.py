import keras.models
from sklearn.externals import joblib
import trainer.predictor as predictor

model = keras.models.load_model("model.hdf5")
scaler = joblib.load("scaler.save")

print(predictor.predict(model, [1], scaler))
print(predictor.predict(model, [0.9], scaler))
print(predictor.predict(model, [0.8], scaler))
print(predictor.predict(model, [0.7], scaler))
print(predictor.predict(model, [0.6], scaler))
print(predictor.predict(model, [0.5], scaler))
print(predictor.predict(model, [0.4], scaler))
print(predictor.predict(model, [0.3], scaler))
print(predictor.predict(model, [0.2], scaler))
print(predictor.predict(model, [0.1], scaler))
print(predictor.predict(model, [0], scaler))