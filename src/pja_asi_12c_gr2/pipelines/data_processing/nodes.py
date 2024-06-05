import logging
from typing import Tuple, Dict
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

    Raises:
        KeyError: If 'Attack' or 'Defense' columns are not in the DataFrame.
        Exception: For any other exceptions that occur during the calculation.
    """
    logger = create_error_logger()
    try:
        if "Attack" not in pokemons.columns or "Defense" not in pokemons.columns:
            raise KeyError("'Attack' or 'Defense' column is missing from the DataFrame")

        # Example coefficient
        pokemons["CustomFeature"] = pokemons["Attack"] / (pokemons["Defense"] + 1)
        return pokemons
    except KeyError as e:
        logger.error("KeyError in calculate_custom_feature: %s", e)
        raise
    except Exception as e:
        logger.error("Error in calculate_custom_feature: %s", e)
        raise


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
        logger.error("Data preparation error: Missing column '%s'", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during DataFrame preparation: %s", e)
        raise
