import pandas as pd
from sklearn import preprocessing


def pre_processing():
    # Load data and review content
    iris_data = pd.read_csv("resources/iris.csv")
    print(iris_data.head())

    # Use label encoder to convert string to numeric values for target variable
    label_encoder = preprocessing.LabelEncoder()
    iris_data['Species'] = label_encoder.fit_transform(iris_data['Species'])
    print(iris_data.head())

    # Convert input to numpy array considering keras input format
    np_iris = iris_data.to_numpy()


if __name__ == '__main__':
    pre_processing()
