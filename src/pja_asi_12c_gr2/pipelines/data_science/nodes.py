from typing import Dict, Any
import logging
from kedro.io import DataCatalog
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def create_error_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def machine_learning(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    logger = create_error_logger()
    try:
        # Creating a pipeline to transform data before applying the classifier
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier(random_state=0)),
            ]
        )
        clf.fit(x_train, y_train)
        return clf
    except ValueError as e:
        logger.error(f"Model training error: {e}")
        raise


def evaluate_model(
    x_test: pd.DataFrame, y_test: pd.Series, classifier: Pipeline
) -> Dict[str, Any]:
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    conf_matrix = confusion_matrix(y_test, y_pred)

    evaluation_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),  # konwersja na listę dla obsługi JSON
    }

    return evaluation_results


def release_model(catalog: DataCatalog, evaluation_results: dict, classifier):
    logger = create_error_logger()
    try:
        catalog.save("evaluation_results", evaluation_results)
    except IOError as e:
        logger.error(f"IOError: {e}")
        raise
