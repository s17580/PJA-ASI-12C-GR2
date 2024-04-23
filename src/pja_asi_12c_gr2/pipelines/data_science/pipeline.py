from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    machine_learning,
    evaluate_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=machine_learning,
                inputs=["x_train", "x_val", "y_train", "y_val", "preprocessor"],
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
