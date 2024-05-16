import logging
from typing import Tuple, Dict
# from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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



def preprocess_pokemons(
    prepared_pokemons: pd.DataFrame,
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """Preprocesses the Pokemon data.

    Args:
        prepared_pokemons: A DataFrame containing the prepared Pokemon data.

    Returns:
        Tuple[pd.DataFrame, ColumnTransformer]: A tuple containing:
            * The preprocessed DataFrame.
            * The fitted ColumnTransformer for scaling and encoding features.
    """
    logger = create_error_logger()
    try:
        categorical_features = ["Type 1", "Type 2"]
        numeric_features = prepared_pokemons.select_dtypes(
            include=["int64", "float64"]
        ).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )
        preprocessed_pokemons = prepared_pokemons
        return preprocessed_pokemons, preprocessor
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise