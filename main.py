import tensorflow as tf

from src.file_handler import load_data
from src.model import shallow_relu_adam, deep_relu_adam

x_train, x_valid, y_train, y_valid = load_data()

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16)
valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(16)

model = deep_relu_adam()

model.fit(train_data, batch_size=16, epochs=20, verbose=1, validation_data=valid_data)

