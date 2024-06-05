from kedro.pipeline import Pipeline, node
from .nodes import (
    split_data,
    train_model,
    evaluate_model,
    # generate_synthetic_data,
    # retrain_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_pokemons", "params:split_data"],
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
            # node(
            #     func=generate_synthetic_data,
            #     inputs=["real_data", "params:synthetic_data.num_samples"],
            #     outputs="synthetic_data",
            #     name="generate_synthetic_data_node",
            # ),
            # node(
            #     func=retrain_model,
            #     inputs=[
            #         "real_data",
            #         "synthetic_data",
            #         "params:machine_learning.logistic_regression",
            #     ],
            #     outputs=["retrained_model", "retraining_results"],
            #     name="retrain_model_node",
            # ),
        ]
    )
