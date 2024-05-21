import logging
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import pandas as pd

def create_error_logger() -> logging.Logger:
    """
    Creates a logger to record errors during pipeline execution.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger

def split_data(
    prepared_pokemons: pd.DataFrame, params: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits the prepared Pokemon data into training, validation, and test sets.

    Args:
        prepared_pokemons: A DataFrame containing the prepared Pokemon data.
        params: A dictionary containing the splitting parameters (test_size, val_size, random_state).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            A tuple containing the following DataFrames and Series in the order:
                * x_train: The features for the training set.
                * x_val: The features for the validation set.
                * x_test: The features for the testing set.
                * y_train: The labels for the training set.
                * y_val: The labels for the validation set.
                * y_test: The labels for the testing set.
    """
    logger = create_error_logger()
    try:
        x = prepared_pokemons
        y = prepared_pokemons["Legendary"]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=params["test_size"],
            stratify=y,
            random_state=params["random_state"],
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_test,
            y_test,
            test_size=params["val_size"],
            stratify=y_test,
            random_state=params["random_state"],
        )
        return x_train, x_val, x_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise
