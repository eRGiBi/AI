import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import os

# Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file(
    "fsns.tfrec",
    "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")

dataset = tf.data.TFRecordDataset(filenames=[fsns_test_file])
# print("\n".join(map(str, dataset)))

raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

print(parsed.features.feature['image/text'])

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)

for line in dataset.take(5):
    print(line.numpy())

files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

# for i, line in enumerate(lines_ds):
#     print()
#     print(line.numpy())
#     print(i)

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

for line in titanic_lines.take(10):
    print(line.numpy())


def survived(line):
    return tf.not_equal(tf.strings.substr(line, 0, 1), "0")


survivors = titanic_lines.skip(1).filter(survived)

for line in survivors.take(10):
    print(line.numpy())

titanic_file = tf.keras.utils.get_file("train.csv",
                                       "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

df = pd.read_csv(titanic_file)
print(df.head())

titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived")

with tf.io.gfile.GFile("titanicbatches.csv", 'w') as f:
    # for line in titanic_batches:
    #     print(line)
    #     f.write(str(line.numpy()))

    for feature_batch, label_batch in titanic_batches.take(1):

        print("'survived': {}".format(label_batch))
        print("features:")
        for key, value in feature_batch.items():
            print("  {!r:20s}: {}".format(key, value))

        titanic_types = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string,
                         tf.string, tf.string]
        dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types, header=True)

        for line in dataset.take(10):
            print([item.numpy() for item in line])

# Creates a dataset that reads all of the records from two CSV files, each with
# four float columns which may have missing values.

record_defaults = [999, 999, 999, 999]
dataset = tf.data.experimental.CsvDataset("missing.csv", record_defaults)
dataset = dataset.map(lambda *items: tf.stack(items))
print(dataset.take(2))

# for line in dataset:
#   print(line.numpy())

flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)
print(flowers_root)

for item in flowers_root.glob("*"):
    print(item.name)

list_ds = tf.data.Dataset.list_files(str(flowers_root / '*/*'))

for f in list_ds.take(5):
    print(f.numpy())


def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label


labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(3):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())

