import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, GRU, LSTM, SimpleRNN
from sklearn.preprocessing import StandardScaler


def buildmodel(n_a, layer):

    x = Input((None, 1))

    if layer == "GRU":
        y = GRU(n_a)(x)
    elif layer == "LSTM":
        y = LSTM(n_a)(x)
    else:
        y = SimpleRNN(n_a)(x)

    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=[x], outputs=[y])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'mean_squared_error'])
    return model


def loaddata(summaryfile, scaler, padvalue):
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
    x = np.full((summarydata.shape[0], maxestimates, 1), padvalue, np.float32)
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


def avg_probability(x):
    y_avg = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        count = 0
        total = 0
        for j in range(x.shape[1]):
            if x[i, j, 0] == -1:
                continue
            total += x[i, j, 0]
            count += 1
        y_avg[i] = total / count

    return y_avg


def predict(m, xlist, scaler):
    x = np.full((1, len(xlist), 1), -1, np.float32)
    for i in range(len(xlist)):
        x[0, i, 0] = xlist[i]

    x.shape = (len(xlist), 1)
    scaler.transform(x, copy=None)
    x.shape = (1, len(xlist), 1)

    return m.predict(x)
