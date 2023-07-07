import nltk
# The next downloads run just one time
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def process():
    spam_data = pd.read_csv('resources/Spam-Classification.csv')
    print(spam_data.head())

    # Separate feature and target data
    spam_classes_raw = spam_data["CLASS"]
    spam_messages = spam_data["SMS"]

    # Build a TF-IDF vectorizer model
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)

    # Transform feature input in TF-IDF
    tfidf = vectorizer.fit_transform(spam_messages)
    # Convert TF-IDF to numpy array
    tfidf_array = tfidf.toarray()

    # Build a label encoder for target variable to convert string to numeric values
    label_encoder = preprocessing.LabelEncoder()
    spam_classes = label_encoder.fit_transform(spam_classes_raw)

    # Convert target to one-hot encoding vector
    spam_classes = tf.keras.utils.to_categorical(spam_classes, 2)
    print("TF-IDF Matrix shape", tfidf.shape)
    print("One-hot encoding shape", spam_classes.shape)

    x_train, x_test, y_train, y_test = train_test_split(tfidf_array, spam_classes, test_size=0.10)

    # Setup hyperparameters for building the model
    nb_classes = 2
    n_hidden = 32

    model = tf.keras.models.Sequential()

    model.add(keras.layers.Dense(n_hidden, input_shape=(x_train.shape[1], ), name='Hidden-Layer-1', activation='relu'))
    model.add(keras.layers.Dense(n_hidden, name='Hidden-Layer-2', activation='relu'))
    model.add(keras.layers.Dense(nb_classes, name='Output-Layer', activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Make it verbose so we can see the progress
    verbose = 1

    # Setup hyperparameters for training
    batch_size = 256
    epochs = 10
    validation_split = 0.2

    print("\nTraining progress:\n-----------------------------")

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split)

    print("\nAccuracy during the training:\n---------------------------")
    pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
    plt.title("Accuracy improvements with epoch")
    plt.show()

    print("\nEvaluation against test dataset:\n---------------------------")
    model.evaluate(x_test, y_test)

    # Predict for multiple samples using batch processing

    # Convert input into IF-IDF vector using the same vectorizer model
    predict_tfidf = vectorizer.transform(["FREE entry to a fun contest", "Yup I will come over"]).toarray()

    print(predict_tfidf.shape)

    # Predict using model
    prediction = np.argmax(model.predict(predict_tfidf), axis=1)
    print("Prediction output: ", prediction)
    print("Prediction classes are: ", label_encoder.inverse_transform(prediction))


# Custom tokenizer to remove stopwords and use lemmatization
def custom_tokenizer(string):
    lemmatizer = WordNetLemmatizer()
    # split string as tokens
    tokens = nltk.word_tokenize(string)
    # filter for stopwords
    nostop = list(filter(lambda token: tokens not in stopwords.words('english'), tokens))
    # perform lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in nostop]
    return lemmatized


if __name__ == '__main__':
    process()
