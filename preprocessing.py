from keras.utils import np_utils

from keras.datasets import mnist


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # set numeric type to float32 from uint8
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Normalize value to [0, 1]
    x_train /= 255
    x_test /= 255

    # Transform labels to one-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Reshape the dataset into 4D array
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_data()

