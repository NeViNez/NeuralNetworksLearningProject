import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
import keras

def neuralNetwork():
    def genData(size=500):
        data = np.random.rand(size, 2) * 2 - 1
        label = np.zeros([size, 1])

        for i, p in enumerate(data):
            if p[0] * p[1] >= 0:
                label[i] = 1.
            else:
                label[i] = 0.

        div = round(size * 0.8)
        train_data = data[:div, :]
        test_data = data[div:, :]
        train_label = label[:div, :]
        test_label = label[div:, :]
        return (train_data, train_label), (test_data, test_label)

    def drawResults(data, label, prediction):
        p_label = np.array([round(x[0]) for x in prediction])
        plt.scatter(data[:, 0], data[:, 1], s=30, c=label[:, 0], cmap=mclr.ListedColormap(['red', 'blue']))
        plt.scatter(data[:, 0], data[:, 1], s=10, c=p_label, cmap=mclr.ListedColormap(['red', 'blue']))
        plt.grid()
        plt.show()

    (train_data, train_label), (test_data, test_label) = genData()

    # NN structure
    model = keras.models.Sequential([
        keras.layers.Dense (32),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2)
    ])

    # Loss function definition
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Adjusting learning rates. Defining optimization algorithms and metrics
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Learning procedure run
    history = model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=50)

    # Catching loss and accuracy metrics history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # Loss graph plots
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Accuracy graph plots
    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    # Test data evaluation and results printing
    results = model.evaluate(test_data, test_label)
    print(f"NN Test Results: {results}")
    # Binary classifying results show
    all_data = np.vstack((train_data, test_data))
    all_label = np.vstack((train_label, test_label))
    pred = model.predict(all_data)
    drawResults(all_data, all_label, pred)