import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

# for elem in dataset:
# print(elem.numpy())

it = iter(dataset)
print(next(it).numpy())

print(dataset.reduce(0, lambda state, value: state + value).numpy())

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1)

print(dataset1.element_spec)

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
     tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

print(dataset2.element_spec)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

print(dataset3.element_spec)

# Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

print(dataset4.element_spec)

# Use value_type to see the type of value represented by the element spec
print(dataset4.element_spec.value_type)

dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))

for x in dataset4:
    print(x)

train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images / 255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))


# for data in dataset:
#     print(data)

def count(stop):
    i = 0
    while i < stop:
        yield i
        i += 1


ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes=(), )

for count_batch in ds_counter.repeat().batch(10).take(10):
    print(count_batch.numpy())


def gen_series():
    i = 0
    while True:
        size = np.random.randint(0, 10)
        yield i, np.random.normal(size=(size,))
        i += 1


for i, series in gen_series():
    print(i, ":", str(series))
    if i > 5:
        break

ds_series = tf.data.Dataset.from_generator(
    gen_series,
    output_types=(tf.int32, tf.float32),
    output_shapes=((), (None,)))

# for x in ds_series:
#     print(x)

ds_series_batch = ds_series.shuffle(20).padded_batch(10)

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())

x = [i for i in range(10)]
x = tf.data.Dataset.from_tensor_slices(x)
for i in x:
    print(i)
print(x.padded_batch(2))
for i in x.padded_batch(2):
    print(i)

# Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec",
                                         "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
# for i in fsns_test_file:
#     print(i)
print(fsns_test_file)
dataset = tf.data.TFRecordDataset(filenames=[fsns_test_file])


