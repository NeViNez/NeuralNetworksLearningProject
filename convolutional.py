import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers

import var7

DIM = 50

def doConv():
    # prepare data
    (x_raw, y_raw) = var7.gen_data(size=10000, img_size=DIM)

    x_train = np.expand_dims(x_raw, -1)

    y_train = np.zeros((x_train.shape[0], ))
    for i, label in enumerate(y_raw):
        if label == 'One':
            n = 0
        elif label == 'Two':
            n = 1
        elif label == 'Three':
            n = 2
        else:
            raise ValueError('wrong label')

        y_train[i] = n

    y_train = keras.utils.to_categorical(y_train, 3)

    # build model
    model = keras.Sequential(
        [
            keras.Input(shape=(DIM, DIM, 1)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(3, activation="softmax"),
        ]
    )


    print(model.summary())

    batch_size = 32
    epochs = 6
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save('model')
