import logging
from typing import Dict, Any, Tuple
import pandas as pd
from kedro.framework.context import KedroContext
from pja_asi_12c_gr2.pipelines.data_science.nodes import train_model, evaluate_model

def select_best_model(
    x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    preprocessor: Any, params: Dict[str, Any]
) -> Any:
    logger = create_error_logger()

    try:
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
    except Exception as e:
        logger.error(f"Error in select_best_model: {e}")
        raise

def create_error_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger

def release_model(classifier):
    logger = create_error_logger()
    try:
        context = KedroContext()
        catalog = context.catalog
        evaluation_results = {"status": "success", "message": "Model evaluated successfully"}
        catalog.save("evaluation_results", evaluation_results)
        catalog.save("classifier", classifier)
    except IOError as e:
        logger.error(f"IOError: {e}")
        raise
