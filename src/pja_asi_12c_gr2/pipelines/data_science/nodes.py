import logging
import pandas as pd
from kedro.io import DataCatalog
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any

def create_error_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger

def get_classifier(classifier_type: str, params: Dict[str, Any]) -> ClassifierMixin:
    classifiers = {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "SVC": SVC,
        "GradientBoostingClassifier": GradientBoostingClassifier
    }
    if classifier_type not in classifiers:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # Poprawiona obsługa None i konwersja typów
    for key, value in list(params.items()):
        if isinstance(value, str):
            if value == 'None':
                params[key] = None
            elif value.isdigit():
                params[key] = int(value)

    params = {key: value for key, value in params.items() if key != 'classifier_type'}
    return classifiers[classifier_type](**params)

def machine_learning(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, preprocessor: ColumnTransformer, params: Dict[str, Any]) -> Pipeline:
    logger = create_error_logger()
    try:
        classifier = get_classifier(params['classifier_type'], params)
        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        clf.fit(x_train, y_train)
        return clf
    except Exception as e:
        logger.error(f"Failed to train classifier: {e}")
        raise

def evaluate_model(x_test: pd.DataFrame, y_test: pd.Series, classifier: Pipeline) -> Dict[str, Any]:
    logger = create_error_logger()
    try:
        y_pred = classifier.predict(x_test)
        evaluation_results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='binary'),
            "recall": recall_score(y_test, y_pred, average='binary'),
            "f1_score": f1_score(y_test, y_pred, average='binary'),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        return evaluation_results
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def release_model(catalog: DataCatalog, evaluation_results: dict, classifier):
    logger = create_error_logger()
    try:
        catalog.save("evaluation_results", evaluation_results)
    except IOError as e:
        logger.error(f"IOError: {e}")
        raise
