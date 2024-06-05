import logging
from typing import Tuple, Dict
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


def calculate_custom_feature(pokemons: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column with the calculated coefficient to the Pokemon DataFrame.

    Args:
        pokemons: Raw Pokemon DataFrame.

    Returns:
        DataFrame: Pokemon DataFrame with 'CustomFeature' column added.
    """
    # Example coefficient
    pokemons["CustomFeature"] = pokemons["Attack"] / (pokemons["Defense"] + 1)
    return pokemons


def prepare_pokemons(pokemons: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Prepares the Pokemon DataFrame for further analysis.

    Args:
        pokemons: The raw Pokemon DataFrame.

    Returns:
        A tuple containing:
          * DataFrame: The Pokemon data with missing "Type 2" values filled and the
                       "Name" column dropped.
          * Dict: A dictionary containing additional metadata about the prepared data:
              - "columns": A list of column names in the prepared DataFrame.
              - "data_type": A string indicating the type of the data ("prepared_pokemons").

    Raises:
        KeyError: If a required column ("Type 2") is missing from the DataFrame.
        Exception: For any other unexpected errors during data preparation.
    """

    logger = create_error_logger()
    try:
        # Fill missing 'Type 2' values
        pokemons["Type 2"] = pokemons["Type 2"].fillna("None")
        # Drop the 'Name' column
        prepared_pokemons = pokemons.drop(labels=["Name"], axis=1)
        prepared_pokemons["CustomFeature"] = (
            prepared_pokemons["Attack"] + prepared_pokemons["Defense"]
        ) / 2
        return prepared_pokemons, {
            "columns": prepared_pokemons.columns.tolist(),
            "data_type": "prepared_pokemons",
        }
    except KeyError as e:
        logger.error(f"Data preparation error: Missing column '{e}'")
        raise  # Re-raise the exception
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error during DataFrame preparation: {e}")
        raise


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
        # Preprocessing
        preprocessed_pokemons = prepared_pokemons
        return preprocessed_pokemons, preprocessor
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise
