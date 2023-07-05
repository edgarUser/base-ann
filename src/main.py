import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def pre_processing():
    # Load data and review content
    iris_data = pd.read_csv("resources/iris.csv")
    print(iris_data.head())

    # Use label encoder to convert string to numeric values for the target variable
    label_encoder = preprocessing.LabelEncoder()
    iris_data['Species'] = label_encoder.fit_transform(iris_data['Species'])
    print(iris_data.head())

    # Convert input to numpy array considering keras input
    np_iris = iris_data.to_numpy()

    # Separate input and target variables
    x_data = np_iris[:, 0:4]
    y_data = np_iris[:, 4]

    print("\nFeatures before scaling :\n------------------------------------")
    print(x_data[:5, :])
    print("\nTarget before scaling :\n------------------------------------")
    print(y_data[:5])

    # Create a scaler model that is fit on the input data
    scaler = StandardScaler().fit(x_data)
    x_data = scaler.transform(x_data)

    # Convert target variable as a one-hot-encoding array
    y_data = tf.keras.utils.to_categorical(y_data, 3)

    print("\nFeatures after scaling :\n------------------------------------")
    print(x_data[:5, :])
    print("\nTarget after one-hot-encoding :\n------------------------------------")
    print(y_data[:5, :])

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Number of classes in the target variable
    nb_classes = 3

    # Create a sequential model in keras
    model = tf.keras.Sequential()

    # Add the first hidden layer (number of nodes, number of input variables, logical name, activation function)
    model.add(keras.layers.Dense(128, input_shape=(4,), name='Hidden-Layer-1', activation='relu'))

    # Add second hidden layer
    model.add(keras.layers.Dense(128, name='Hidden-Layer-2', activation='relu'))

    # Add an output layer with softmax activation
    model.add(keras.layers.Dense(nb_classes, name='Output-Layer', activation='softmax'))

    # Compile the model with loss and metrics
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model metadata
    model.summary()

    # Make it verbose so we can see the progress
    verbose = 1

    # Setup Hyper Parameters for training

    # Set Batch size
    batch_size = 16
    # Set number of epochs
    epochs = 10
    # Set validation split. 20% of the training data will be used for validation
    # after each epoch
    validation_split = 0.2

    print("\nTraining Progress:\n------------------------------------")

    # Fit the model. This will perform the entire training cycle, including
    # forward propagation, loss computation, backward propagation and gradient descent.
    # Execute for the specified batch sizes and epoch
    # Perform validation after each epoch

    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split)

    # Plot accuracy of the model after epoch
    pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
    plt.title("Accuracy improvements with Epoch")
    plt.show()

    # Evaluate the model against the test dataset and print results
    print("\nEvaluation against Test Dataset :\n------------------------------------")
    model.evaluate(x_test, y_test)

    # Saving a model
    model.save("resources/iris_save")

    # Loading a Model
    loaded_model = keras.models.load_model("resources/iris_save")

    # Print model summary
    loaded_model.summary()

    # Raw prediction data
    prediction_input = [[6.6, 3., 4.4, 1.4]]

    # Scale prediction data with the same scaling model
    scaled_input = scaler.transform(prediction_input)

    # Get raw prediction probabilities
    raw_prediction = model.predict(scaled_input)
    print("Raw Prediction Output (Probabilities) :", raw_prediction)

    # Find prediction
    prediction = np.argmax(raw_prediction)
    print("Prediction is ", label_encoder.inverse_transform([prediction]))


if __name__ == '__main__':
    pre_processing()
