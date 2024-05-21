from ast import Param
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
    """
    Creates and configures a logger for error handling.

    Returns:
        logging.Logger: A configured logger object set to log errors.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def get_classifier(classifier_type: str, params: Dict[str, Any]) -> ClassifierMixin:
    """
    Gets an instance of a classifier based on the provided type and parameters.

    Args:
        classifier_type: String specifying the type of classifier ("DecisionTreeClassifier", "RandomForestClassifier", "SVC", or "GradientBoostingClassifier").
        params: A dictionary containing parameters to initialize the classifier.

    Returns:
        ClassifierMixin: An instance of the specified classifier.

    Raises:
        ValueError: If an unsupported classifier type is provided.
        TypeError: If there's a type mismatch in the provided parameters.
    """
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
    """
    Creates and trains a machine learning pipeline.

    Args:
        x_train: The training features.
        x_val: The validation features.
        y_train: The training labels.
        y_val: The validation labels.
        preprocessor: The preprocessor for feature engineering.
        params: A dictionary of parameters for the classifier.

    Returns:
        Pipeline: The trained pipeline.

    Raises:
        Exception: If training fails.
    """
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
    """
    Evaluates a trained model on test data and logs metrics using wandb.

    Args:
        x_test: The test features.
        y_test: The test labels.
        classifier: The trained pipeline.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation metrics.

    Raises:
        ValueError: If an error occurs during evaluation.
        KeyError: If metric logging fails.
        OSError: If wandb initialization fails.
    """
    logger = create_error_logger()
    try:
        y_pred = classifier.predict(x_test)
        y_probas = classifier.predict_proba(x_test) if hasattr(classifier.named_steps['classifier'], "predict_proba") else None
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Initialize wandb session
        # os.chdir("C:")
        os.chdir(os.path.abspath("."))
        wandb.init(project="PJA-ASI-12C-GR2", dir=os.path.abspath("."))

        # Log metrics in wandb
        wandb.log(
            {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        )
        
        # classifier = DecisionTreeClassifier()
        # classifier = SVC()
        classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
        classifier = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
        classifier = RandomForestClassifier(n_estimators=75, max_depth=5, random_state=0)
        # classifier = GradientBoostingClassifier()

        table = wandb.Table(data = x_test, columns = x_test.columns)
        wandb.log({"X_test" : table})

        # wandb.sklearn.plot_classifier(evaluate_model,x_test,y_test,y_pred,classifier,
        #                               is_binary = True,
        #                               model_name = 'RandomForest')
       

        # log artifacts
        raw_data = wandb.Artifact('raw_data', type='dataset')
        raw_data.add_dir('data/01_raw')
        wandb.log_artifact(raw_data)

        # training_dataset = wandb.Artifact('training_dataset', type='dataset')
        # training_dataset.add_file('data/02_itermediate/prepared_pokemons.csv')
        # wandb.log_artifact(training_dataset)

        # model_artifact = wandb.Artifact('model', type='model')
        # model_artifact.add_file('data/05_model_output/release_model.pk/2024-05-21T00.06.12.507Z/release_model.pk')
        # wandb.log_artifact(model_artifact)

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
    """Saves evaluation results and the model to the Kedro DataCatalog.

    Args:
        catalog: The Kedro DataCatalog instance.
        evaluation_results: A dictionary containing evaluation metrics.
        classifier: The trained model pipeline.

    Raises:
        IOError: If an error occurs while saving the data.
    """
    logger = create_error_logger()
    try:
        catalog.save("evaluation_results", evaluation_results)
    except IOError as e:
        logger.error(f"IOError: {e}")
        raise
