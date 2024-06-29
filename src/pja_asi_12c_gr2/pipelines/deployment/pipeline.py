from kedro.pipeline import Pipeline, node
from .nodes import select_best_model


def create_pipeline(**kwargs):
    """Creates a Kedro pipeline for model selection and release.

    This pipeline performs the following steps:

    1. select_best_model (Optional): Evaluates multiple models (if provided) on
                                     the validation set and selects the best
                                     performing one based on a chosen metric.
    2. release_model: Releases the selected model or the default model if no
                      selection is made.

    Args:
        **kwargs: Additional keyword arguments that can be passed to Kedro nodes,
            including:
                * params:machine_learning: A dictionary containing:
                    * models: A list of dictionaries, each defining a model with
                              its name, type, and parameters. (optional, required if
                              `select_best_model` is used).
                    * selection_metric: The metric used for model selection (optional,
                              defaults to accuracy if `select_best_model` is used).

    Returns:
        Pipeline: A Kedro pipeline object defining the model selection and release workflow.
    """
    return Pipeline(
        [
            node(
                func=select_best_model,
                inputs=[
                    "x_train",
                    "x_val",
                    "x_test",
                    "y_train",
                    "y_val",
                    "y_test",
                    "params:machine_learning",
                ],
                outputs="best_model",
                name="select_best_model_node",
                tags=["optional"],
            ),
        ]
    )
