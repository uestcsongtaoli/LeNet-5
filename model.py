from keras.models import Sequential
from keras.layers import Conv2D, Dense, AveragePooling2D, Activation, Flatten


def build_model(input_layer):
    model = Sequential()

    """
    另外一种写法
    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1),
                     input_shape=input_layer, padding="same"))
    model.add(Activation("relu"))
    """

    # C1 Convolutional Layer
    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation="relu",
                     input_shape=input_layer, padding="same"))

    # S2 Pooling Layer
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid"))

    # C3 Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation="relu",
                     padding="valid"))

    # S4 Pooling Layer
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

    # C5 Fully Connected Convolutional Layer
    model.add(Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation="relu",
                     padding="valid"))
    model.add(Flatten())
    # FC6 Fully Connected Layer
    model.add(Dense(84, activation="relu"))

    # Output Layer with softmax activation
    model.add(Dense(10, activation="softmax"))

    return model
