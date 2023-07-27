from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD


def shallow_relu_adam():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(52160,)))
    model.add(Dropout(.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model


def simple_sigmoid_sgd():
    model = Sequential()
    model.add(Dense(8, activation='sigmoid', input_shape=(52160,)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

    return model


def deep_relu_adam():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(52160,)))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model
