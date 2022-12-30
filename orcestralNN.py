import numpy as np
from tensorflow import keras
from keras import layers

import var7
import datetime

DIM = 50

def doOrcestral():
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

    # Собственный callback
    class SelectedEpochSave(keras.callbacks.Callback):
        MODEL_NAME_PREFIX = "conv_epoch"
        EPOCHS = (1, 3, 4, 5)

        def on_epoch_end(self, epoch, logs=None):
            if epoch + 1 in self.EPOCHS:
                self.model.save(self.compile_model_name(epoch))

        def compile_model_name(self, epoch: int) -> str:
            return f"{datetime.datetime.now().date()}_{self.MODEL_NAME_PREFIX}_{epoch + 1}.h5"

    print(model.summary())

    batch_size = 32
    epochs = 6
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
              callbacks=[SelectedEpochSave()])
    model.save('model')
