import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, GRU, LSTM, SimpleRNN
from sklearn.preprocessing import StandardScaler

Features = 1


def buildmodel(n_a, layer):

    x = Input((None, Features))

    if layer == "GRU":
        y = GRU(n_a)(x)
    elif layer == "LSTM":
        y = LSTM(n_a)(x)
    elif layer == "2xSimpleRNN":
        y = SimpleRNN(n_a, return_sequences=True)(x)
        y = SimpleRNN(n_a)(y)
    else:
        y = SimpleRNN(n_a)(x)

    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=[x], outputs=[y])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'mean_squared_error'])
    return model


def loaddata(summaryfile, scaler, padvalue):
    summarydata = pd.read_csv(summaryfile, sep=',', header=None)
    allresponses = pd.read_csv('responsedata.csv', sep=',', header=None)

    xlists_estimates = []
    xlists_times = []
    xlists_hascomment = []
    y = np.zeros((summarydata.shape[0], 1))
    maxestimates = 0
    for i in range(summarydata.shape[0]):
        responses = allresponses[(allresponses[0] == summarydata.iloc[i, 0]) & allresponses[2].notnull()]
        estimates = responses[2].values.tolist()
        xlists_estimates.append(estimates)
        xlists_times.append(responses[1].values.tolist())
        xlists_hascomment.append(responses[4].notnull().tolist())
        y[i][0] = (1 if summarydata.iloc[i, 5] == 1 else 0)
        if len(estimates) > maxestimates:
            maxestimates = len(estimates)

    # We put our padding at the start rather than the end of the data,
    # to make it easier to learn.
    x = np.full((summarydata.shape[0], maxestimates, Features), padvalue, np.float32)
    for i in range(summarydata.shape[0]):
        for j in range(len(xlists_estimates[i])):
            feature = 0
            x[i, j+maxestimates-len(xlists_estimates[i]), feature] = xlists_estimates[i][j]
            feature += 1

            # Timestamp feature
            # x[i, j+maxestimates-len(xlists_estimates[i]), feature] = summarydata.iloc[i, 2] - xlists_times[i][j]
            # feature += 1

            # Question length feature
            # question = summarydata.iloc[i, 7]
            # x[i, j + maxestimates - len(xlists_estimates[i]), feature] = len(question) if type(question) == "str" else 0
            # feature += 1

            # Has Comment feature
            # x[i, j + maxestimates - len(xlists_estimates[i]), feature] = 1 if xlists_hascomment[i][j] else 0
            # feature += 1

    x.shape = (summarydata.shape[0] * maxestimates, Features)

    if isinstance(scaler, bool) and scaler:
        scaler = StandardScaler()
        scaler.fit(x)

    if isinstance(scaler, StandardScaler):
        x = scaler.transform(x, copy=None)

    x.shape = (summarydata.shape[0], maxestimates, Features)

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


def predict(m, xlist_estimates, scaler):
    x = np.full((1, len(xlist_estimates), Features), -1, np.float32)
    for i in range(len(xlist_estimates)):
        feature = 0
        x[0, i, feature] = xlist_estimates[i]
        feature += 1

        # Timestamp feature
        # x[0, i, feature] = duetime - xlist_times[i]
        # feature += 1

        # Question length feature
        # x[0, i, feature] = len(question)
        # feature += 1

        # Has Comment feature
        # x[0, i, feature] = 1 if xlist_hascomment[i] else 0
        # feature += 1

    x.shape = (len(xlist_estimates), Features)
    scaler.transform(x, copy=None)
    x.shape = (1, len(xlist_estimates), Features)

    return m.predict(x)
