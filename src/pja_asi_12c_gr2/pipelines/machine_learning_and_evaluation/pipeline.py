from kedro.pipeline import Pipeline, node
from .nodes import (
    train_model,
    evaluate_model,
    release_model,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "x_train",
                    "x_val",
                    "y_train",
                    "y_val",
                    "preprocessor",
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
                func=release_model,
                inputs=["evaluation_results", "classifier"],
                outputs=None,
                name="release_model_node",
            ),
        ]
    )
