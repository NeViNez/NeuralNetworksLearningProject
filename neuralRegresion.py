import random
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly_express as px


def dataset_generation(dataset_path, function, repeats=30000):
    dataset_list = list()
    for idx in range(repeats):
        if idx % 1000 == 0:
            print(f"{idx} Generating...")
        sample_list = list()
        for x in np.arange(3.0, 10.0, 0.01):
            e = random.randrange(-150000, 150000, 1) / 1000000
            sample_value = function(x) + e
            sample_list.append(sample_value)
        dataset_list.append(sample_list)
    df = pd.DataFrame(dataset_list)
    df.to_csv(dataset_path, index=False)


def linear_regression_fit(linear_dataset_path, repeats):
    df = pd.read_csv(linear_dataset_path)   # Чтение сгенерированного датасета
    x_list = np.arange(3.0, 10.0, 0.01)     # Делаем начальный список с training labels
    # Зачем делаем датафрейм?
    x_df = pd.DataFrame(x_list).T           # Транспоз, чтобы ты заплакала
    # Этот метод работает медленно. И вообще идея с отдельным созданием меток не нравится, все должно быть в
    # датасете изначально
    for i in range(repeats - 1):
        temp = pd.DataFrame(np.arange(3.0, 10.0, 0.01)).T
        x_df = pd.concat([x_df, temp])

    print(x_df)

    # Нормализация для ленивых. Этот слой нужен только чтобы работало, я пытался сделать нормалайз сам, но не завелось
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_df)

    # Смысл в том, что его здесь нет. Почему один нейрон? Потому что так было в руководстве...
    # Поидее это потому что регрессия линейная и x -> y, но кто бы это знал...
    regr_model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(1)
    ])
    regr_model.build()
    regr_model.summary()
    regr_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    # Да оно не должно работать... почему работает???
    history = regr_model.fit(
        x_df,
        np.array(df),
        epochs=70,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.1)

    # Еще и графики какие-то красивые рисует, жесть...
    print("Abobus complete")
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    a = px.line(hist)
    a.show()


def normalization(numbers_list):
    maxnum = max(numbers_list)
    for idx in range(len(numbers_list)):
        numbers_list[idx] /= maxnum
    return numbers_list


def sine_function(x):
    return np.sin(x)


def linear_function(x):
    return x - 3
