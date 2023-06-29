from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from src.file_handler import load_data

x_train, y_train, x_valid, y_valid = load_data()

model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=(7,)))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(x_valid, y_valid))

