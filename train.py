from keras.losses import categorical_crossentropy
from model import build_model
from preprocessing import get_data
from keras.utils import multi_gpu_model
from utils import save_model
from keras.callbacks import TensorBoard
import argparse

FLAGS = None


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = get_data()
    input_layer = (28, 28, 1)

    model = build_model(input_layer)
    model.summary()

    # Parallel
    parallel_model = multi_gpu_model(model, gpus=2)

    # Compile the model
    parallel_model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=["accuracy"])

    # 记录训练过程
    tensorboard = TensorBoard(log_dir='./logs/adam_relu', histogram_freq=0,
                              write_graph=True, write_images=False)
    # Train the model
    history = parallel_model.fit(x=x_train,
                                 y=y_train,
                                 epochs=10,
                                 batch_size=128,
                                 validation_data=(x_test, y_test),
                                 callbacks=[tensorboard],
                                 verbose=1)
    # Save the model
    model_name = "LeNet-5.h5"
    save_model(model=parallel_model, model_name=model_name)



    # Visualize the Training Process, tensorboard is better.
    """
    tensorboard = TensorBoard(log_dir=f'logs/{time()}')
    model.fit(
        callback=[tensorboard]
    )
    """



