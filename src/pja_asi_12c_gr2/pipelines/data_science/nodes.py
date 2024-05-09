from typing import Dict, Any
import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from kedro.io import DataCatalog
import pandas as pd
import wandb


def create_error_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def get_classifier(classifier_type: str, params: Dict[str, Any]) -> ClassifierMixin:
    classifiers = {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "SVC": SVC,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    }
    if classifier_type not in classifiers:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # None handling and types conversion
    logger = create_error_logger()
    for key, value in list(params.items()):
        if isinstance(value, str):
            if value == "None":
                params[key] = None
            elif value.isdigit():
                params[key] = int(value)
    try:
        params = {
            key: value for key, value in params.items() if key != "classifier_type"
        }
        return classifiers[classifier_type](**params)
    except TypeError as e:
        logger.error(f"Type conversion error for classifier parameters: {e}")
        raise


def machine_learning(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    preprocessor: ColumnTransformer,
    params: Dict[str, Any],
) -> Pipeline:
    logger = create_error_logger()
    try:
        classifier = get_classifier(params["classifier_type"], params)
        clf = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        clf.fit(x_train, y_train)
        return clf
    except Exception as e:
        logger.error(f"Failed to train classifier: {e}")
        raise


def evaluate_model(
    x_test: pd.DataFrame, y_test: pd.Series, classifier: Pipeline
) -> Dict[str, Any]:
    logger = create_error_logger()
    try:
        y_pred = classifier.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Initialize wandb session
        os.chdir("C:")
        wandb.init(project="actions", dir=os.path.abspath("."))

        # Log metrics in wandb
        wandb.log(
            {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    except (ValueError, KeyError, OSError) as e:
        logger.error(f"Model evaluation error: {e}")
        raise


def release_model(catalog: DataCatalog, evaluation_results: dict, classifier):
    logger = create_error_logger()
    try:
        catalog.save("evaluation_results", evaluation_results)
    except IOError as e:
        logger.error(f"IOError: {e}")
        raise
