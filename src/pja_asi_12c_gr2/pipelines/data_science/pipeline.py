from kedro.pipeline import Pipeline, node
from .nodes import (
    split_data,
    train_model,
    evaluate_model,
    generate_synthetic_data,
    retrain_model,
)


def create_pipeline(**kwargs):
    """Creates a Kedro pipeline for training, evaluating, and potentially retraining a
    Pokemon classification model.

    This pipeline performs the following steps:

    1. split_data: Splits the prepared Pokemon data into training, validation, and test sets.
    2. train_model: Trains a machine learning model (specified in the 'params:machine_learning'
                   configuration) using the training and validation data, potentially using
                   AutoML if 'params:autoML' is True.
    3. evaluate_model: Evaluates the trained model on the test data and calculates relevant
                       metrics.
    4. generate_synthetic_data: Generates synthetic Pokemon data based on the real dataset.
    5. retrain_model: Retrains the initially trained model (or a new model, if specified)
                      using both real and synthetic data.

    Args:
        **kwargs: Additional keyword arguments that can be passed to Kedro nodes, including:
            * params:split_data: Parameters for data splitting.
            * params:machine_learning.<model_name>: Parameters for the chosen model.
            * params:autoML: A boolean indicating whether to use AutoML for model training.
            * params:synthetic_data.num_samples: Number of synthetic samples to generate.

    Returns:
        Pipeline: A Kedro Pipeline object defining the model training, evaluation, and
                  potential retraining workflow.
    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["prepared_pokemons", "params:split_data"],
                outputs=["x_train", "x_val", "x_test", "y_train", "y_val", "y_test"],
                name="split_data",
            ),
            node(
                func=train_model,
                inputs=[
                    "x_train",
                    "x_val",
                    "y_train",
                    "y_val",
                    "params:machine_learning.decision_tree",
                    "params:autoML",
                ],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "x_test",
                    "y_test",
                    "classifier",
                    "params:autoML",
                ],
                outputs="evaluation_results",
                name="evaluate_model_node",
            ),
            node(
                func=generate_synthetic_data,
                inputs=["real_data", "params:synthetic_data.num_samples"],
                outputs="synthetic_data",
                name="generate_synthetic_data_node",
            ),
            node(
                func=retrain_model,
                inputs=[
                    "real_data",
                    "synthetic_data",
                    "params:machine_learning.logistic_regression",
                ],
                outputs=["retrained_model", "retraining_results"],
                name="retrain_model_node",
            ),
        ]
    )
