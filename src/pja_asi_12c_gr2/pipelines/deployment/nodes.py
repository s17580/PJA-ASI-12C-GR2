import wandb
import logging
from typing import Dict, Any
import pandas as pd
from kedro.framework.context import KedroContext
from pja_asi_12c_gr2.pipelines.data_science.nodes import train_model, evaluate_model
import joblib
import os


def select_best_model(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    params: Dict[str, Any],
) -> Any:
    """Trains and compares two models to select the best one based on F1 score.

    This function performs the following steps:

    1. Trains a model using AutoGluon, leveraging automated machine learning (autoML = True).
    2. Trains a model without AutoGluon, using the hyperparameters specified in the 'params' dictionary.
    3. Evaluates both models on the test data using the 'evaluate_model' function.
    4. Compares the F1 scores of the two models.
    5. Returns the model with the higher F1 score.

    Args:
        x_train (pd.DataFrame): Features for the training set.
        x_val (pd.DataFrame): Features for the validation set.
        x_test (pd.DataFrame): Features for the test set.
        y_train (pd.Series): Labels for the training set.
        y_val (pd.Series): Labels for the validation set.
        y_test (pd.Series): Labels for the test set.
        params (Dict[str, Any]): A dictionary containing model hyperparameters
                                 (e.g., 'classifier_type', 'max_depth', etc.).

    Returns:
        Any: The best performing model object (either an AutoGluon predictor or a
             scikit-learn pipeline with a fitted classifier).

    Raises:
        Exception: If an error occurs during model training or evaluation. This will
                   be logged for further investigation.
    """
    logger = create_error_logger()
    wandb.init(project="PJA_SUML_11c_gr4")

    try:
        # Train with auto-gluon
        autoML_params = params.copy()
        autoML_params["autoML"] = True
        autoML_model = train_model(
            x_train, x_val, y_train, y_val, autoML_params, autoML=True
        )
        autoML_results = evaluate_model(x_test, y_test, autoML_model, autoML=True)

        # Train without auto-gluon
        regular_params = params.copy()
        regular_params["autoML"] = False
        regular_model = train_model(
            x_train, x_val, y_train, y_val, regular_params, autoML=False
        )
        regular_results = evaluate_model(x_test, y_test, regular_model, autoML=False)

        # Compare models and the best one
        if autoML_results["f1"] > regular_results["f1"]:
            model_path = os.getenv("MODEL_PATH", "data/06_model_output/best_model.pkl")
            joblib.dump(autoML_model, model_path)
            wandb.log({"best_model": "AutoML", "autoML_results": autoML_results})
            return autoML_model
        else:
            model_path = os.getenv("MODEL_PATH", "data/06_model_output/best_model.pkl")
            joblib.dump(regular_model, model_path)
            wandb.log({"best_model": "Regular", "regular_results": regular_results})
            return regular_model
    except Exception as e:
        logger.error("Error in select_best_model: %s", e)
        raise


def create_error_logger() -> logging.Logger:
    """
    Creates a logger to record errors during pipeline execution.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger
