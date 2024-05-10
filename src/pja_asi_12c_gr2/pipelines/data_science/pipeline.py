from kedro.pipeline import Pipeline, node, pipeline

from .nodes import machine_learning, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for training and evaluating a model.

    This pipeline performs the following steps:

    1. machine_learning: Trains a machine learning model using the training and
                         validation data.
    2. evaluate_model: Evaluates the trained model on the test data,
                       logs metrics, and returns a dictionary containing the
                       evaluation results.

    Args:
        **kwargs: Additional keyword arguments that can be passed to Kedro nodes,
                  such as:
                      * params:machine_learning.<classifier_name>: A dictionary of
                               parameters for the specific machine learning algorithm.
                      * params:split_data.<parameters>: A dictionary of parameters
                               for data splitting.
    Returns:
        Pipeline: A Kedro Pipeline object defining the model training and
                  evaluation workflow.
    """
    return pipeline(
        [
            node(
                func=machine_learning,
                inputs=[
                    "x_train",
                    "x_val",
                    "y_train",
                    "y_val",
                    "preprocessor",
                    "params:machine_learning.decision_tree",
                ],
                outputs="classifier",
                name="machine_learning",
            ),
            node(
                func=evaluate_model,
                inputs=["x_test", "y_test", "classifier"],
                outputs="release_model",
            ),
        ]
    )
