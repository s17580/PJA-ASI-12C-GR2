import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pickle
from sklearn.base import is_classifier


def transform_data(pokemons):
    # df = load_data()
    pokemons["Type 2"] = pokemons["Type 2"].fillna("None")  # Filling missing values

    # Features and Labels
    # Dropping "Name" and "Legendary" columns
    X = pokemons.drop(labels=["Name", "Legendary"], axis=1)

    # Target variable
    y = pokemons["Legendary"]

    # Encoding categorical features "Type 1" and "Type 2"
    categorical_features = ["Type 1", "Type 2"]
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

    # Creating a column transformer to transform categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    # Creating a pipeline to transform data before applying the classifier
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=0)),
        ]
    )

    # Splitting the dataset into training and testing sets
    x_train1, x_test1, y_train1, y_test1 = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    return x_train1, x_test1, y_train1, y_test1, clf


def machine_learning(x_train1, x_test1, y_train1, y_test1, clf):
    clf.fit(x_train1, y_train1)
    return clf


# def evaluate_model1(x_test1, y_test1, classifier):
#         y_pred = classifier.predict(x_test1)

#         accuracy = accuracy_score(y_test1, y_pred)
#         precision = precision_score(y_test1, y_pred, average="binary")
#         recall = recall_score(y_test1, y_pred, average="binary")
#         f1 = f1_score(y_test1, y_pred, average="binary")
#         conf_matrix = confusion_matrix(y_test1, y_pred)

#         print("Accuracy:", accuracy)
#         print("Precision:", precision)
#         print("Recall:", recall)
#         print("F1 Score:", f1)
#         print("Confusion Matrix:\n", conf_matrix)

# def release_model(pokemons):
#     with open("pokemon_classifier.pkl", "wb") as f:
#         pickle.dump((transform_data, is_classifier), f)
