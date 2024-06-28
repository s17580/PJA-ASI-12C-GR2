import logging
from typing import Dict, Any, Tuple
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from ctgan import CTGAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


def create_error_logger() -> logging.Logger:
    """
    Creates a logger to record errors during pipeline execution.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Makes all column names in the DataFrame unique by appending numeric suffixes to duplicates.

    This function checks for duplicate column names in the provided DataFrame. If any duplicates are found,
    it modifies them by adding an underscore followed by a sequential number (_1, _2, etc.).
    The original column names are left unmodified if they are already unique.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with unique column names.

    Raises:
        Exception: If an unexpected error occurs during column renaming (logged for debugging).
    """
    logger = create_error_logger()
    try:
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                dup + "_" + str(i) if i != 0 else dup for i in range(sum(cols == dup))
            ]
        df.columns = cols
        return df
    except Exception as e:
        logger.error("Error in making columns unique: %s", e)
        raise


def autogluon_train(
    train_data: pd.DataFrame, val_data: pd.DataFrame, params: Dict[str, Any]
) -> TabularPredictor:
    """Trains a tabular classification or regression model using AutoGluon.

    This function performs the following steps:

    1. Ensures column names in both the training and validation datasets are unique.
    2. Initializes a TabularPredictor from AutoGluon, specifying the target column.
    3. Fits the predictor on the training data, using the validation data for hyperparameter tuning.
       The `presets` parameter (defaulting to "medium_quality_faster_train") controls the trade-off
       between model quality and training speed.

    Args:
        train_data: A pandas DataFrame containing the training data.
        val_data: A pandas DataFrame containing the validation data.
        params: A dictionary containing parameters for AutoGluon training, including:
            * target_column (str): The name of the column containing the target variable.
            * presets (str, optional): The AutoGluon preset to use. Defaults to "medium_quality_faster_train".

    Returns:
        TabularPredictor: The fitted AutoGluon predictor.

    Raises:
        Exception: If an error occurs during model training or data preparation.
    """
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
        logger.error("Error during AutoGluon training: %s", e)
        raise


def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses test data for model inference.

    This function performs the following steps on the input test DataFrame:
    1. Checks if the 'Legendary' column exists, and if so, casts it to an integer type and creates a new column 'Legendary_1'.
    2. Ensures that all column names in the DataFrame are unique using `make_column_names_unique`.
    3. If after step 2 the 'Legendary_1_1' column does not exist, it is created as a copy of 'Legendary_1'.

    Args:
        df (pd.DataFrame): The raw test data as a pandas DataFrame.

    Returns:
        pd.DataFrame: The preprocessed test data DataFrame with unique column names and potential adjustments for the 'Legendary' column.

    Raises:
        Exception: If any error occurs during preprocessing (logged with the traceback for debugging purposes).
    """
    logger = create_error_logger()
    try:
        if "Legendary" in df.columns:
            df["Legendary_1"] = df["Legendary"].astype(int)
        df = make_column_names_unique(df)

        if "Legendary_1_1" not in df.columns:
            df["Legendary_1_1"] = df["Legendary_1"]
        return df
    except Exception as e:
        logger.error("Error in preprocess_test_data: %s", e)
        raise


def split_data(
    prepared_pokemons: pd.DataFrame, params: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits the prepared Pokemon data into training, validation, and test sets.

    This function performs two consecutive train-test splits:

    1. The first split divides the data into a training set (1 - `test_size`) and a temporary set
       for testing and validation combined (`test_size`).
    2. The second split divides the temporary set into equal validation and test sets.

    Both splits use stratification to maintain the same distribution of the target variable
    ('Legendary') in each set.

    Args:
        prepared_pokemons (pd.DataFrame): The prepared Pokemon DataFrame containing features and the target variable.
        params (Dict[str, float]): A dictionary of parameters for data splitting, including:
            * test_size (float): The proportion of the dataset to include in the combined test and validation set.
            * val_size (float): The proportion of the combined test and validation set to include in the validation set.
            * random_state (int): The seed used by the random number generator for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: A tuple containing the following DataFrames and Series in the order:
            * x_train: The features for the training set.
            * x_val: The features for the validation set.
            * x_test: The features for the testing set.
            * y_train: The labels for the training set.
            * y_val: The labels for the validation set.
            * y_test: The labels for the testing set.

    Raises:
        Exception: If an error occurs during the splitting process (e.g., invalid parameter values).
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
        logger.error("Error during data splitting: %s", e)
        raise


def train_model(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    params: Dict[str, Any],
    autoML: bool = False,
) -> Any:
    """Trains a machine learning model on Pokemon data.

    This function either trains a DecisionTreeClassifier with specified hyperparameters
    or uses AutoGluon for automated machine learning.

    Args:
        x_train (pd.DataFrame): The feature matrix for training data.
        x_val (pd.DataFrame): The feature matrix for validation data.
        y_train (pd.Series): The target variable for training data.
        y_val (pd.Series): The target variable for validation data.
        params (Dict[str, Any]): Dictionary containing model hyperparameters. If using AutoGluon,
            it should include:
                * target_column (str): The name of the target column.
                * presets (str, optional): The AutoGluon preset to use. Defaults to 'medium_quality_faster_train'.
            If not using AutoGluon, it should include:
                * max_depth (int, optional): The maximum depth of the decision tree. Defaults to 10.
                * min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 2.
                * random_state (int, optional): The seed used by the random number generator for reproducibility. Defaults to 0.
        autoML (bool, optional): If True, use AutoGluon for training. Otherwise, train a DecisionTreeClassifier. Defaults to False.

    Returns:
        Any: A trained model object (either Pipeline with DecisionTreeClassifier or TabularPredictor from AutoGluon).

    Raises:
        Exception: If an error occurs during preprocessing, model training, or AutoGluon setup.
    """
    logger = create_error_logger()
    try:
        categorical_features = ["Type 1", "Type 2"]
        numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise
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
        logger.error("Failed to train model: %s", e)
        raise


def evaluate_model(
    x_test: pd.DataFrame, y_test: pd.Series, model: Any, autoML: bool = False
) -> Dict[str, Any]:
    """Evaluates a trained machine learning model on test data.

    This function performs model evaluation on the provided test data and returns
    a dictionary of performance metrics. If `autoML` is True, additional preprocessing
    is performed on the test data before evaluation.

    Args:
        x_test (pd.DataFrame): The feature matrix for the test data.
        y_test (pd.Series): The true labels for the test data.
        model (Any): The trained model (either a scikit-learn Pipeline or AutoGluon TabularPredictor).
        autoML (bool, optional): If True, indicates that an AutoGluon model is being used, and additional
                                preprocessing is required for the test data. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing the following evaluation metrics:
            * accuracy (float): The accuracy of the model.
            * precision (float): The weighted precision of the model.
            * recall (float): The weighted recall of the model.
            * f1 (float): The weighted F1-score of the model.

    Raises:
        ValueError: If an error occurs during evaluation, such as mismatched data shapes.
        KeyError: If a metric calculation fails due to an invalid label.
        OSError: If there's an issue with the operating system during evaluation (e.g., file access error).
    """
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
        logger.error("Model evaluation error: %s", e)
        raise


def generate_synthetic_data(
    real_data: pd.DataFrame, num_samples: int, max_attempts: int = 5
) -> pd.DataFrame:
    """Generates synthetic data using the CTGAN model from SDV.

    Args:
        real_data (pd.DataFrame): The original DataFrame used for training the model.
        num_samples (int): The number of synthetic samples to generate.

    Returns:
        pd.DataFrame: The generated synthetic data.

    Raises:
        ValueError: If `num_samples` is not a positive integer.
        Exception: If there are issues with the data format or if the CTGAN model fails to fit or sample.
    """
    logger = create_error_logger()
    attempt = 0

    while attempt < max_attempts:
        try:
            logger.info(f"Attempt {attempt+1}: Generating synthetic data...")
            # Prepare the model
            discrete_columns = real_data.select_dtypes(
                include=["object"]
            ).columns.tolist()
            model = CTGAN(
                embedding_dim=128,
                generator_dim=(256, 256, 256),
                discriminator_dim=(256, 256, 256),
                batch_size=500,
                epochs=300,
            )

            # Fit the model
            model.fit(real_data, discrete_columns)

            # Generate synthetic data
            synthetic_data = model.sample(num_samples)

            # Ensure the synthetic data has the same columns as the real data
            if synthetic_data.shape[1] != real_data.shape[1]:
                logger.error(
                    f"Shape mismatch: synthetic data has {synthetic_data.shape[1]} columns, expected {real_data.shape[1]}"
                )
                raise ValueError(
                    f"Shape mismatch: synthetic data has {synthetic_data.shape[1]} columns, expected {real_data.shape[1]}"
                )

            logger.info(f"Synthetic data columns: {synthetic_data.columns}")

            # Verify if synthetic data contains more than one class in the target column
            if synthetic_data["target"].nunique() > 1:
                return synthetic_data
            else:
                logger.warning(
                    "Synthetic data contains only one class in the target column"
                )
                attempt += 1
        except Exception as e:
            logger.error("Error during synthetic data generation: %s", e)
        finally:
            attempt += 1

    raise ValueError(
        "Failed to generate synthetic data with more than one class in the target column after multiple attempts"
    )


def retrain_model(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[Any, Dict[str, float]]:
    """Retrains a logistic regression model using a combination of real and synthetic data.

    This function performs the following steps:

    1. Concatenates the real and synthetic data into a combined dataset.
    2. Splits the combined data into features (X) and the target variable (y), assuming the target is named "target".
    3. Preprocesses the features using a ColumnTransformer that applies StandardScaler to numeric features and OneHotEncoder to categorical features.
    4. Splits the preprocessed data into training (80%) and testing (20%) sets.
    5. Retrains a logistic regression model using the provided parameters on the training data.
    6. Evaluates the retrained model on the testing data using the F1 score.

    Args:
        real_data (pd.DataFrame): The original DataFrame containing the real data.
        synthetic_data (pd.DataFrame): The DataFrame containing the synthetic data.
        params (Dict[str, Any]): A dictionary of parameters for the logistic regression model.

    Returns:
        Tuple[Any, Dict[str, float]]: A tuple containing:
            * The retrained model object (LogisticRegression).
            * A dictionary with the F1 score as the evaluation result.

    Raises:
        Exception: If an error occurs during data preprocessing, model training, or evaluation (logged for debugging).
    """
    logger = create_error_logger()

    try:
        combined_data = pd.concat([real_data, synthetic_data])
        if "target" not in combined_data.columns:
            logger.error("The 'target' column is missing from the combined data")
            raise KeyError("The 'target' column is missing from the combined data")

        # Check for NaNs in the target column
        if combined_data["target"].isna().sum() > 0:
            logger.error("The 'target' column contains NaN values")
            raise ValueError("The 'target' column contains NaN values")

        # Verify if combined data contains more than one class in the target column
        if combined_data["target"].nunique() <= 1:
            logger.error("Combined data contains only one class in the target column")
            raise ValueError(
                "Combined data contains only one class in the target column"
            )

        # Fill missing 'Type 2' values
        combined_data["Type 2"] = combined_data["Type 2"].fillna("None")

        # Drop the 'Name' column
        combined_data.drop(labels=["Name"], axis=1)

        # Fill NaNs in synthetic_data
        combined_data.fillna(0, inplace=True)

        # Split data into features and target
        X = combined_data.drop("target", axis=1)
        y = combined_data["target"]

        # Preprocess the combined data
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        # categorical_features = ["Type 1", "Type 2", "Legendary"]
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
        logger.error("Error during retraining the model: %s", e)
        raise
