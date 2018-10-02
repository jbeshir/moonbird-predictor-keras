import numpy as np
import predictor

# Hyperparameters
n_a = 4
epochs = 100
layer = "SimpleRNN"
padvalue = -1
dense_layers = 1

embeddings_index = predictor.loadembeddings('glove.6B.50d.txt')
print('Found %s word vectors.' % len(embeddings_index))

# Build model
model = predictor.buildmodel(n_a, layer, dense_layers)
print(model.summary())

# Load training data and train
Xtrain, Ytrain, scaler = predictor.loaddata('summarydata-train.csv', embeddings_index, True, padvalue)
model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=128)

# Load CV data and evaluate
Xcv, Ycv, _ = predictor.loaddata('summarydata-cv.csv', embeddings_index, scaler, padvalue)
print(model.evaluate(Xcv, Ycv, batch_size=128))

Xcv_avg, _, _ = predictor.loaddata('summarydata-cv.csv', embeddings_index, False, padvalue)
Ycv_avg = predictor.avg_probability(Xcv_avg)
print("Naive averaging MSE:", np.mean(np.square(Ycv_avg - Ycv), axis=0))

# Save
# model.save("model.hdf5")
# joblib.dump(scaler, "scaler.save")
