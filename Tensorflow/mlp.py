import os
import tensorflow as tf
import keras
from keras import layers


class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units
        print("Linear im being __init__")

    def build(self, input_shape):
        print("Linear im being built")
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        print(self.w)
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
        print(self.b)

    def call(self, inputs):
        print("Linear im being called")
        print(inputs)
        return tf.matmul(inputs, self.w) + self.b


class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        print("MLPBlock im being called")
        x = self.linear_1(inputs)
        print(x)
        x = tf.nn.relu(x)
        print(x)
        x = self.linear_2(x)
        print(x)
        x = tf.nn.relu(x)
        print(x)
        return self.linear_3(x)


mlp = MLPBlock()
print(tf.ones(shape=(3, 64)))
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print(y)
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))
