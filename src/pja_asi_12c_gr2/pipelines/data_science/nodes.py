import logging
from typing import Dict, Any, Tuple
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from kedro.framework.context import KedroContext
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# from sdv.tabular import CTGAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


def create_error_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            dup + "_" + str(i) if i != 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df


def autogluon_train(
    train_data: pd.DataFrame, val_data: pd.DataFrame, params: Dict[str, Any]
) -> TabularPredictor:
    logger = create_error_logger()
    try:
        train_data = make_column_names_unique(train_data)
        val_data = make_column_names_unique(val_data)

        predictor = TabularPredictor(label=params["target_column"]).fit(
            train_data,
            tuning_data=val_data,
            presets=params.get("presets", "medium_quality_faster_train"),
        )
        return predictor
    except Exception as e:
        logger.error(f"Error during AutoGluon training: {e}")
        raise


def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    if "Legendary" in df.columns:
        df["Legendary_1"] = df["Legendary"].astype(int)
    df = make_column_names_unique(df)

    if "Legendary_1_1" not in df.columns:
        df["Legendary_1_1"] = df["Legendary_1"]
    return df


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


def train_model(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    preprocessor: Any,
    params: Dict[str, Any],
    autoML: bool = False,
) -> Any:
    logger = create_error_logger()
    try:
        if autoML:
            train_data = pd.concat([x_train, y_train.rename("Legendary")], axis=1)
            val_data = pd.concat([x_val, y_val.rename("Legendary")], axis=1)
            train_data = train_data.rename(columns={"Legendary": "Legendary_1"})
            val_data = val_data.rename(columns={"Legendary": "Legendary_1"})
            params["target_column"] = "Legendary_1"
            predictor = autogluon_train(train_data, val_data, params)
            return predictor
        else:
            classifier = DecisionTreeClassifier(
                max_depth=params.get("max_depth", 10),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=params.get("random_state", 0),
            )
            clf = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
            clf.fit(x_train, y_train)
            return clf
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise


def evaluate_model(
    x_test: pd.DataFrame, y_test: pd.Series, model: Any, autoML: bool = False
) -> Dict[str, Any]:
    logger = create_error_logger()
    try:
        if autoML:
            x_test = preprocess_test_data(x_test)
            y_pred = model.predict(x_test)
        else:
            y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    except (ValueError, KeyError, OSError) as e:
        logger.error(f"Model evaluation error: {e}")
        raise


'''
def generate_synthetic_data(real_data: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """
    Generates synthetic data using the CTGAN model from SDV.

    Args:
        real_data: The original DataFrame.
        num_samples: Number of synthetic samples to generate.

    Returns:
        DataFrame: The synthetic data.
    """
    model = CTGAN()
    model.fit(real_data)
    synthetic_data = model.sample(num_samples)
    return synthetic_data



def retrain_model(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[Any, Dict[str, float]]:
    """
    Retrains the model using a combination of real and synthetic data.

    Args:
        real_data: The original real data DataFrame.
        synthetic_data: The synthetic data DataFrame.
        params: Parameters for the model.

    Returns:
        A tuple containing:
          * The retrained model.
          * A dictionary with evaluation results.
    """
    logger = create_error_logger()

    try:
        # Combine real and synthetic data
        combined_data = pd.concat([real_data, synthetic_data])

        # Split data into features and target
        X = combined_data.drop("target", axis=1)
        y = combined_data["target"]

        # Preprocess the combined data
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )

        X_preprocessed = preprocessor.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y, test_size=0.2, random_state=42
        )

        # Retrain the model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        return model, {"f1": f1}
    except Exception as e:
        logger.error(f"Error in retrain_model: {e}")
        raise
    '''
