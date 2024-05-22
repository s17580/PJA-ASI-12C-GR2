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
    Dodaje kolumnę z obliczanym współczynnikiem do DataFrame Pokemonów.

    Args:
        pokemons: Surowy DataFrame Pokemonów.

    Returns:
        DataFrame: DataFrame Pokemonów z dodaną kolumną 'CustomFeature'.
    """
    pokemons['CustomFeature'] = pokemons['Attack'] / (pokemons['Defense'] + 1)  # Przykładowy współczynnik
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
