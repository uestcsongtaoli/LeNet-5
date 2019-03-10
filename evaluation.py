from keras.models import load_model
from preprocessing import load_data

if __name__ == "__main__":
    file_path = ""
    (x_train, y_train), (x_test, y_test) = load_data(file_path)
    # Load model
    model_path = ""
    model = load_model(model_path)

    # Evaluate the model
    test_score = model.evaluate(x_test, y_test)
    # y_pred = model.predict_classes(x_test)
    print(f"Test loss: {test_score[0]:.4f}, accuracy {test_score[1] * 100:.2f}%.")

    # Plot
