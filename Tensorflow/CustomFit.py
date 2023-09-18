import tensorflow as tf
import keras
import numpy as np
#
#
# dataset = tf.data.Dataset.range(2)
# print(dataset)
# for element in dataset:
#   print(element)
# dataset = tf.data.Dataset.range(2)
# iterator = iter(dataset)
# print(iterator.get_next())
#
# print(iterator.get_next())
#
#
#
# class CustomModel(keras.Model):
#     def train_step(self, data):
#         # Unpack the data. Its structure depends on your model and
#         # on what you pass to `fit()`.
#         x, y = data
#         print("x:", x)
#         print("y:", y)
#
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             # Compute the loss value
#             # (the loss function is configured in `compile()`)
#             print(y_pred)
#             loss = self.compute_loss(y=y, y_pred=y_pred)
#             print(loss)
#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         print(trainable_vars)
#         gradients = tape.gradient(loss, trainable_vars)
#         print(gradients)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Update metrics (includes the metric that tracks the loss)
#         for metric in self.metrics:
#             if metric.name == "loss":
#                 metric.update_state(loss)
#             else:
#                 metric.update_state(y, y_pred)
#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}
#
#
# # Construct and compile an instance of CustomModel
# inputs = keras.Input(shape=(32,))
# print(inputs)
# outputs = keras.layers.Dense(1)(inputs)
# print(outputs)
# model = CustomModel(inputs, outputs)
# model.compile(optimizer="adam", loss="mse", metrics=["mae"])
#
# # Just use `fit` as usual
# x = np.random.random((1000, 32))
# y = np.random.random((1000, 1))
# model.fit(x, y, epochs=3)

class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compute_loss(y=y, y_pred=y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):

        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics

        # for metric in self.metrics:
        #     if metric.name == "loss":
        #         metric.update_state(loss)
        #     else:
        #         metric.update_state(y, y_pred, sample_weight=sample_weight)

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mae_metric]


# Construct an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)

# We don't passs a loss or metrics here.
model.compile(optimizer="adam")

# Just use `fit` as usual -- you can use callbacks, etc.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)