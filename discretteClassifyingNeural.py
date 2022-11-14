import numpy as np
import tensorflow as tf
import keras

def discretteNeural():
    binaryList = [0, 1]

    h = []

    for a in binaryList:
        for b in binaryList:
            for c in binaryList:
                answer = a and (b or c)
                h.append([np.array([a, b, c]), answer])

    h = np.array(h)
    print(h)

    abc = tf.convert_to_tensor([i[0] for i in h], dtype=tf.float32)
    ans = tf.convert_to_tensor([i[1] for i in h], dtype=tf.float32)
    print(abc)
    print(ans)

    model = keras.Sequential(
        [
            keras.layers.Dense(1024, activation="relu", name="layer1"),
            keras.layers.Dense(512, activation="relu", name="layer2"),
            keras.layers.Dense(256, name="layer3"),
            keras.layers.Dense(128, name="layer4"),
            keras.layers.Dense(2, name="layer5"),
        ]
    )

    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(abc, ans, epochs=90, batch_size=8)