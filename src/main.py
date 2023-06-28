import pandas as pd


def pre_processing():
    iris_data = pd.read_csv("iris.csv")
    print(iris_data.head())


if __name__ == '__main__':
    pre_processing()
