import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import flwr as fl

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 逆向きにする
# train_images = train_images[::-1]
# train_labels = train_labels[::-1]

# 指定した数削除
for i in range(30000):
  train_images = np.delete(train_images, 0, axis=0)
  train_labels = np.delete(train_labels, 0)

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dense(10)
  ])

model.compile(optimizer="adam",
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

class CifarClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    model.fit(train_images, train_labels, epochs=10)
    return model.get_weights(), len(train_images), {}

  def evaluate(self, parameters, config):
    model.set_weights(parameters)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    return test_loss, len(test_images), {"accuracy": float(test_acc)}

fl.client.start_numpy_client(server_address="192.168.9.10:8080", client=CifarClient())