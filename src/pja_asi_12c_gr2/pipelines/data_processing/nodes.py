import logging
from kedro.io import DataCatalog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def create_error_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def preprocess_pokemons(
    pokemons: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    ColumnTransformer,
]:
    logger = create_error_logger()
    try:
        pokemons["Type 2"] = pokemons["Type 2"].fillna("None")  # Filling missing values

        # Features and Labels
        # Dropping "Name" column
        preprocessed_pokemons = pokemons.drop(labels=["Name"], axis=1)

        # Encoding categorical features "Type 1" and "Type 2"
        categorical_features = ["Type 1", "Type 2"]
        numeric_features = preprocessed_pokemons.select_dtypes(
            include=["int64", "float64"]
        ).columns

        # Creating a column transformer to transform categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )
        return preprocessed_pokemons, preprocessor
    except FileNotFoundError:
        logger.error("Pokemon dataset not found at the specified path.")
        raise  # Re-raise the exception for the pipeline to handle
    except (KeyError, ValueError) as e:
        logger.error(f"Data processing error: {e}")
        raise


def split_data(preprocessed_pokemons):
    # Train dataset
    x = preprocessed_pokemons
    # Target variable
    y = preprocessed_pokemons["Legendary"]
    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=0
    )

    # Splitting the testing dataset into testing and validation sets
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, stratify=y_test, random_state=0
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
