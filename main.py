import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, LSTM

# Hyperparameters
n_a = 5

X = Input((None, 1))
X1 = LSTM(n_a)(X)
Y = Dense(1, activation='softmax')(X1)

model = Model(inputs=[X], outputs=[Y])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def loaddata(summaryfile):
    summarydata = pd.read_csv(summaryfile, sep=',', header=None)
    allresponses = pd.read_csv('responsedata.csv', sep=',', header=None)

    x = []
    y = []
    for i in range(summarydata.shape[0]):
        responses = allresponses[(allresponses[0] == summarydata.iloc[i, 0]) & allresponses[2].notnull()]
        x.append(responses[2].values.tolist())
        y.append(1 if summarydata.iloc[i, 5] == 1 else 0)

    return x, y


# Load training data
Xtrain, Ytrain = loaddata('summarydata-train.csv')
for i in range(100):
    print(Xtrain[i], Ytrain[i])
