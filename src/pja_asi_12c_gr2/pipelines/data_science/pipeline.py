from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    transform_data,
    machine_learning,
    evaluate_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=transform_data,
                inputs="pokemons",
                outputs=["x_train", "x_test", "y_train", "y_test", "clf"],
                name="transform_data_node",
            ),
            node(
                func=machine_learning,
                inputs=["x_train", "x_test", "y_train", "y_test", "clf"],
                outputs="classifier",
                name="machine_learning_node",
            ),
            node(
                func=evaluate_model,
                inputs=["x_test", "y_test", "classifier"],
                outputs="release_model",
            ),
        ]
    )
