import logging
from typing import Dict, Any
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from kedro.framework.context import KedroContext
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


def create_error_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
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
    logger = create_error_logger()
    try:
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
    if 'Legendary' in df.columns:
        df['Legendary_1'] = df['Legendary'].astype(int)
    df = make_column_names_unique(df)

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
    logger = create_error_logger()
    try:
        if autoML:
            train_data = pd.concat([x_train, y_train.rename('Legendary')], axis=1)
            val_data = pd.concat([x_val, y_val.rename('Legendary')], axis=1)
            train_data = train_data.rename(columns={'Legendary': 'Legendary_1'})
            val_data = val_data.rename(columns={'Legendary': 'Legendary_1'})
            params['target_column'] = 'Legendary_1'
            predictor = autogluon_train(train_data, val_data, params)
            return predictor
        else:
            classifier = DecisionTreeClassifier(
                max_depth=params["max_depth"], 
                min_samples_split=params["min_samples_split"], 
                random_state=params["random_state"]
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


def champion_vs_challenger(
    x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    preprocessor: Any, params: Dict[str, Any]
) -> Any:
    # Train with auto-gluon
    autoML_params = params.copy()
    autoML_params['autoML'] = True
    autoML_model = train_model(x_train, x_val, y_train, y_val, preprocessor, autoML_params, autoML=True)
    autoML_results = evaluate_model(x_test, y_test, autoML_model, autoML=True)

    # Train without auto-gluon
    regular_params = params.copy()
    regular_params['autoML'] = False
    regular_model = train_model(x_train, x_val, y_train, y_val, preprocessor, regular_params, autoML=False)
    regular_results = evaluate_model(x_test, y_test, regular_model, autoML=False)

    # Compare
    if autoML_results['f1'] > regular_results['f1']:
        return autoML_model
    else:
        return regular_model


def release_model(evaluation_results: dict, classifier):
    logger = create_error_logger()
    try:
        context = KedroContext()
        catalog = context.catalog
        catalog.save("evaluation_results", evaluation_results)
        catalog.save("classifier", classifier)
    except IOError as e:
        logger.error(f"IOError: {e}")
        raise
