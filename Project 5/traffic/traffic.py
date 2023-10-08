import random

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 50
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for sub_folder in map(str, range(NUM_CATEGORIES)):

        s_path = os.path.join(data_dir, sub_folder)

        # print(s_path)

        # images = filter(lambda filename: filename.lower().endswith(('.pmm')), os.listdir(subfolder_path))
        # images = os.listdir(path)

        for image_file in os.listdir(s_path):
            img = cv2.imread(os.path.join(s_path, image_file))
            img.resize((IMG_WIDTH, IMG_HEIGHT, 3))

            images.append(img)
            labels.append(sub_folder)
    # ds = tf.data.Dataset.from_tensor_slices(np.array(images),np.array(labels))
    # ds = ds.shuffle(buffer_size=len(images))

    # print(images)
    # print(labels)
    # return (ds)
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPool2D(pool_size=(4, 4)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(526, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(266, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),

    ])
    for i in range(50):
        model.add(tf.keras.layers.Dense(random.randint(500, 2000), activation="relu"))

    model.add(tf.keras.layers.Dropout(0.5),)

    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    model.compile(
        # optimizer=tf.keras.optimizers.RMSprop(),
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        # loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    return model


if __name__ == "__main__":
    main()
