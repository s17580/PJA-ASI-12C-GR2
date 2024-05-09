import logging
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


def create_error_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def prepare_pokemons(pokemons: pd.DataFrame) -> pd.DataFrame:
    """Prepares the Pokemon DataFrame for further analysis.

    Args:
        pokemons: The raw Pokemon DataFrame.

    Returns:
        A DataFrame with filled missing values and the "Name" column removed.
    """

    logger = create_error_logger()
    try:
        # Fill missing 'Type 2' values
        pokemons["Type 2"] = pokemons["Type 2"].fillna("None")
        # Drop the 'Name' column
        prepared_pokemons = pokemons.drop(labels=["Name"], axis=1)

        return prepared_pokemons
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
        return preprocessor
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise


def split_data(
    prepared_pokemons: pd.DataFrame, params: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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
