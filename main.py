from keras.models import Model
from keras.layers import Dense, Input, LSTM

# Hyperparameters
n_a = 5

X = Input((None, 1))
x = LSTM(n_a)(X)
Y = Dense(1, activation='softmax')(x)

model = Model(inputs=[X], outputs=[Y])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])