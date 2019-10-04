import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GRU, LSTM, Masking, SimpleRNN, Subtract, Multiply
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from tensorflow.python.lib.io import file_io

Features = 1


def loadembeddings(path):
    embeddings_index = {}

    with file_io.FileIO(path, mode='r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def avgembedding(embeddings_index, s):
    avg_embedding = np.zeros(50)
    count = 0

    if not isinstance(s, str):
        return avg_embedding

    words = s.split(' ')
    for word in words:
        embedding = embeddings_index.get(word.strip(',.-').strip())
        if embedding is not None:
            avg_embedding += embedding
            count += 1

    if count != 0:
        avg_embedding = avg_embedding / count
    return avg_embedding


def buildmodel_defaults(scaler=None):
    # Hyperparameters
    n_a = 4
    epochs = 100
    layer = "LSTM"
    dense_layers = 1
    l1 = 0.0
    l2 = 0.0

    return buildmodel(n_a, layer, dense_layers, l1, l2, scaler)


def buildmodel(n_a, layer, dense_layers, l1, l2, scaler=None):

    x = Input((None, Features))
    y = x
    if scaler is not None:
        mean = K.constant(scaler.mean_, dtype='float32', name='mean', shape=(1, Features))
        scale = K.constant(1 / scaler.scale_, dtype='float32', name='scale', shape=(1, Features))
        y = Subtract()([y, Input(tensor=mean)])
        y = Multiply()([y, Input(tensor=scale)])

    y = Masking(mask_value=0.0)(y)

    if layer == "GRU":
        y = GRU(n_a, kernel_regularizer=l1_l2(l1=l1, l2=l2), bias_regularizer=l1_l2(l1=l1, l2=l2))(y)
    elif layer == "LSTM":
        y = LSTM(n_a, kernel_regularizer=l1_l2(l1=l1, l2=l2), bias_regularizer=l1_l2(l1=l1, l2=l2))(y)
    elif layer == "2xSimpleRNN":
        y = SimpleRNN(n_a, kernel_regularizer=l1_l2(l1=l1, l2=l2), bias_regularizer=l1_l2(l1=l1, l2=l2), return_sequences=True)(y)
        y = SimpleRNN(n_a, kernel_regularizer=l1_l2(l1=l1, l2=l2), bias_regularizer=l1_l2(l1=l1, l2=l2))(y)
    elif layer == "2xLSTM":
        y = LSTM(n_a, kernel_regularizer=l1_l2(l1=l1, l2=l2), bias_regularizer=l1_l2(l1=l1, l2=l2), return_sequences=True)(y)
        y = LSTM(n_a, kernel_regularizer=l1_l2(l1=l1, l2=l2), bias_regularizer=l1_l2(l1=l1, l2=l2))(y)
    else:
        y = SimpleRNN(n_a)(y)

    for i in range(dense_layers-1):
        y = Dense(4, kernel_regularizer=l1_l2(l1=l1, l2=l2), bias_regularizer=l1_l2(l1=l1, l2=l2))(y)

    y = Dense(1, kernel_regularizer=l1_l2(l1=l1, l2=l2), bias_regularizer=l1_l2(l1=l1, l2=l2), activation='sigmoid')(y)

    if scaler is not None:
        model = Model(inputs=[x, mean, scale], outputs=[y])
    else:
        model = Model(inputs=[x], outputs=[y])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def loaddata(summaryfile, responsefile, embeddings_index, scaler):
    with file_io.FileIO(summaryfile, mode='r') as f:
        summarydata = pd.read_csv(f, sep=',', header=None)
    with file_io.FileIO(responsefile, mode='r') as f:
        allresponses = pd.read_csv(f, sep=',', header=None)

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

    padsize = maxestimates

    # We put our padding at the start rather than the end of the data,
    # to make it easier to learn.
    x = np.full((summarydata.shape[0], padsize, Features), np.nan, np.float32)
    for i in range(summarydata.shape[0]):
        for j in range(len(xlists_estimates[i])):
            feature = 0
            x[i, j+padsize-len(xlists_estimates[i]), feature] = xlists_estimates[i][j]
            feature += 1

            # Timestamp feature
            # x[i, j+padsize-len(xlists_estimates[i]), feature] = summarydata.iloc[i, 2] - xlists_times[i][j]
            # feature += 1

            # Question length feature
            # question = summarydata.iloc[i, 7]
            # x[i, j + padsize - len(xlists_estimates[i]), feature] = len(question) if isinstance(question, str) else 0
            # feature += 1

            # Has Comment feature
            # x[i, j + padsize - len(xlists_estimates[i]), feature] = 1 if xlists_hascomment[i][j] else 0
            # feature += 1

            # Question avg wordvec feature
            # question = summarydata.iloc[i, 7]
            # x[i, j + padsize - len(xlists_estimates[i]), feature:feature+50]\
            #     = avgembedding(embeddings_index, question)
            # feature += 50

    x.shape = (summarydata.shape[0] * padsize, Features)

    if isinstance(scaler, bool) and scaler:
        scaler = StandardScaler()
        scaler.fit(x)

    if isinstance(scaler, StandardScaler):
        x = scaler.transform(x, copy=None)
        x = np.nan_to_num(x, copy=False)

    x.shape = (summarydata.shape[0], padsize, Features)

    return x, y, scaler


def avg_probability(x):
    y_avg = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        count = 0
        total = 0
        for j in range(x.shape[1]):
            if np.isnan(x[i, j, 0]):
                continue
            total += x[i, j, 0]
            count += 1
        y_avg[i] = total / count

    return y_avg


def predict(m, xlist_estimates, scaler):
    padsize = len(xlist_estimates)
    x = np.full((1, padsize, Features), np.nan, np.float32)
    for i in range(len(xlist_estimates)):
        feature = 0
        x[0, padsize - len(xlist_estimates) + i, feature] = xlist_estimates[i]
        feature += 1

        # Timestamp feature
        # x[0, padsize - len(xlist_estimates) + i, feature] = duetime - xlist_times[i]
        # feature += 1

        # Question length feature
        # x[0, padsize - len(xlist_estimates) + i, feature] = len(question)
        # feature += 1

        # Has Comment feature
        # x[0, padsize - len(xlist_estimates) + i, feature] = 1 if xlist_hascomment[i] else 0
        # feature += 1

    if scaler is not None:
        x.shape = (padsize, Features)
        x = scaler.transform(x, copy=None)
        x.shape = (1, padsize, Features)

    x = np.nan_to_num(x, copy=False)

    return m.predict(x)
