from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    transform_data,
    machine_learning,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=transform_data,
                inputs="pokemons",
                outputs=["x_train1", "x_test1", "y_train1", "y_test1", "clf"],
                name="transform_data_node",
            ),
            node(
                func=machine_learning,
                inputs=["x_train1", "x_test1", "y_train1", "y_test1", "clf"],
                outputs="machine_learning_node",
            ),
            # node(
            #     func=evaluate_model1,
            #     inputs=["x_test1", "y_test1", "classifier"],
            #     outputs="metrics1",
            #     ),
        ]
    )
