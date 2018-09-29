from sklearn.externals import joblib
import numpy as np
import predictor

# Hyperparameters
n_a = 4
epochs = 100

# Build model
model = predictor.buildmodel(n_a)
print(model.summary())

# Load training data and train
Xtrain, Ytrain, scaler = predictor.loaddata('summarydata-train.csv', None)
model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=128)

# Load CV data and evaluate
Xcv, Ycv, _ = predictor.loaddata('summarydata-cv.csv', scaler)
print(model.evaluate(Xcv, Ycv, batch_size=128))

Ycv_avg = predictor.avg_probability(Xcv)
print("Naive averaging MSE:", np.mean(np.square(Ycv_avg - Ycv), axis=0))

# Save
model.save("model.hdf5")
joblib.dump(scaler, "scaler.save")
