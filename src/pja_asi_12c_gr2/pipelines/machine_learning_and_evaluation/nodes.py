import logging
from typing import Dict, Any
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from kedro.io import DataCatalog

def create_error_logger() -> logging.Logger:
    """
    Creates and configures a logger for error handling.

    Returns:
        logging.Logger: A configured logger object set to log errors.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger

def make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures unique column names in a DataFrame by appending suffixes to duplicates.

    Args:
        df: DataFrame with potential duplicate column names.

    Returns:
        DataFrame with unique column names.
    """
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def autogluon_train(
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    params: Dict[str, Any]
) -> TabularPredictor:
    """
    Trains an AutoML model using AutoGluon.

    Args:
        train_data: Training data with the target column.
        val_data: Validation data with the target column.
        params: Dictionary of parameters for AutoGluon.

    Returns:
        predictor: The trained AutoGluon predictor.
    """
    logger = create_error_logger()
    try:
        # Ensure unique column names
        train_data = make_column_names_unique(train_data)
        val_data = make_column_names_unique(val_data)

        predictor = TabularPredictor(label=params['target_column']).fit(
            train_data, 
            tuning_data=val_data, 
            presets=params.get('presets', 'medium_quality_faster_train')
        )
        return predictor
    except Exception as e:
        logger.error(f"Error during AutoGluon training: {e}")
        raise

def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the test data to match the training data format.

    Args:
        df: Test DataFrame.

    Returns:
        Preprocessed test DataFrame.
    """
    if 'Legendary' in df.columns:
        df['Legendary_1'] = df['Legendary'].astype(int)
    df = make_column_names_unique(df)  # Ensure unique column names

    # Ensure the test data has the same preprocessing steps
    if 'Legendary_1_1' not in df.columns:
        df['Legendary_1_1'] = df['Legendary_1']
    return df

def train_model(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    preprocessor: Any,
    params: Dict[str, Any],
    autoML: bool = False
) -> Any:
    """
    Trains a machine learning model using the specified method.

    Args:
        x_train: The training features.
        x_val: The validation features.
        y_train: The training labels.
        y_val: The validation labels.
        preprocessor: The preprocessor for feature engineering.
        params: A dictionary of parameters for the classifier.
        autoML: Flag to use AutoML (AutoGluon) or standard ML.

    Returns:
        The trained model (Pipeline or TabularPredictor).
    """
    logger = create_error_logger()
    try:
        if autoML:
            train_data = pd.concat([x_train, y_train.rename('Legendary')], axis=1)
            val_data = pd.concat([x_val, y_val.rename('Legendary')], axis=1)
            train_data = train_data.rename(columns={'Legendary': 'Legendary_1'})
            val_data = val_data.rename(columns={'Legendary': 'Legendary_1'})
            params['target_column'] = 'Legendary_1'  # Set target column
            predictor = autogluon_train(train_data, val_data, params)
            return predictor
        else:
            from sklearn.pipeline import Pipeline
            classifier = get_classifier(params["classifier_type"], params)
            clf = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
            clf.fit(x_train, y_train)
            return clf
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise

def evaluate_model(
    x_test: pd.DataFrame, y_test: pd.Series, model: Any, autoML: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a trained model on test data.

    Args:
        x_test: The test features.
        y_test: The test labels.
        model: The trained model (Pipeline or TabularPredictor).
        autoML: Flag to use AutoML (AutoGluon) or standard ML.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation metrics.
    """
    logger = create_error_logger()
    try:
        if autoML:
            x_test = preprocess_test_data(x_test)
            y_pred = model.predict(x_test)
            y_probas = model.predict_proba(x_test)
        else:
            y_pred = model.predict(x_test)
            y_probas = model.predict_proba(x_test) if hasattr(model.named_steps['classifier'], "predict_proba") else None

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

def release_model(evaluation_results: dict, classifier):
    """
    Saves evaluation results and the model to the Kedro DataCatalog.

    Args:
        evaluation_results: A dictionary containing evaluation metrics.
        classifier: The trained model.

    Raises:
        IOError: If an error occurs while saving the data.
    """
    logger = create_error_logger()
    try:
        context = KedroContext()  # Obtain the context
        catalog = context.catalog  # Get the catalog from context
        catalog.save("evaluation_results", evaluation_results)
        catalog.save("classifier", classifier)
    except IOError as e:
        logger.error(f"IOError: {e}")
        raise
