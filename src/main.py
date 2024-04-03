import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# LoadData
def loadData():
    df = pd.read_csv("pokemon.csv", index_col=False)
    return df


# PrepareData
def prepareData():
    df = loadData()

    df = df.drop(labels=["#"], axis=1)
    df["Type 2"] = df["Type 2"].fillna("None")

    x = df.iloc[:, 0:10].values
    y = df.iloc[:, 11].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=0
    )
    # Standarization - TO DO
    sc = StandardScaler()

    return x_train, x_test, y_train, y_test


# MachineLearning
# ModelEvaluation
# ModelRelease
# Run modules
x_train, x_test, y_train, y_test = prepareData()
