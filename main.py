import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, SimpleRNN
from sklearn.preprocessing import StandardScaler


def buildmodel(n_a):
    x = Input((None, 1))
    y = SimpleRNN(n_a)(x)
    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=[x], outputs=[y])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model


def loaddata(summaryfile, scaler):
    summarydata = pd.read_csv(summaryfile, sep=',', header=None)
    allresponses = pd.read_csv('responsedata.csv', sep=',', header=None)

    xlists = []
    y = np.zeros((summarydata.shape[0], 1))
    maxestimates = 0
    for i in range(summarydata.shape[0]):
        responses = allresponses[(allresponses[0] == summarydata.iloc[i, 0]) & allresponses[2].notnull()]
        estimates = responses[2].values.tolist()
        xlists.append(estimates)
        y[i][0] = (1 if summarydata.iloc[i, 5] == 1 else 0)
        if len(estimates) > maxestimates:
            maxestimates = len(estimates)

    # We put our padding at the start rather than the end of the data,
    # to make it easier to learn.
    x = np.full((summarydata.shape[0], maxestimates, 1), -1, np.float32)
    for i in range(summarydata.shape[0]):
        for j in range(len(xlists[i])):
            x[i, j+maxestimates-len(xlists[i]), 0] = xlists[i][j]

    x.shape = (summarydata.shape[0] * maxestimates, 1)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(x)

    scaler.transform(x, copy=None)

    x.shape = (summarydata.shape[0], maxestimates, 1)

    return x, y, scaler


def predict(m, xlist, scaler):
    x = np.full((1, len(xlist), 1), -1, np.float32)
    for i in range(len(xlist)):
        x[0, i, 0] = xlist[i]

    x.shape = (len(xlist), 1)
    scaler.transform(x, copy=None)
    x.shape = (1, len(xlist), 1)

    return m.predict(x)


# Build model
model = buildmodel(2)
print(model.summary())

# Load training data and train
Xtrain, Ytrain, scaler = loaddata('summarydata-train.csv', None)
model.fit(Xtrain, Ytrain, epochs=100, batch_size=128)

# Load CV data and evaluate
Xcv, Ycv, _ = loaddata('summarydata-cv.csv', scaler)
print(model.evaluate(Xcv, Ycv, batch_size=128))

# Visualise
print(predict(model, [0.99], scaler))
print(predict(model, [0.5], scaler))
print(predict(model, [0.01], scaler))
