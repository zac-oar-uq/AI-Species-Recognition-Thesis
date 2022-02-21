from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from os import listdir
from os.path import isfile
import math
from prepare_data import *
from datetime import datetime


def main():
    dataset = Dataset.carabid
    train_gen, val_gen, test_gen = prep_data(dataset, True, 4)

    inputs = keras.layers.Input(shape=(None, 299, 299, 3))

    # Model 1
    base_model1 = InceptionV3(weights="imagenet")
    for layer in base_model1.layers:
        layer._name += "_1"
        layer.trainable = True

    timed_layer = keras.layers.TimeDistributed(base_model1)(inputs)

    lstm_layer = keras.layers.CuDNNLSTM(1000)(timed_layer)
    dense_layer = keras.layers.Dense(1000, activation='relu')(lstm_layer)
    predictions = keras.layers.Dense(291, activation='softmax')(dense_layer)

    model = keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    logdir = "logs/time_distributed" + "_" + str(dataset) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model.fit(train_gen, validation_data=val_gen, callbacks=[tensorboard_callback], epochs=5)
    model.evaluate(test_gen)


if __name__ == "__main__":
    main()

