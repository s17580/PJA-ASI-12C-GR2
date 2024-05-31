from kedro.pipeline import Pipeline, node
from .nodes import select_best_model, release_model

def create_pipeline(**kwargs):
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
                tags=["optional"]
            ),
            node(
                func=release_model,
                inputs=["best_model"],
                outputs=None,
                name="release_model_node",
            ),
        ]
    )
