from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_pokemons,
    split_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_pokemons,
                inputs="pokemons",
                outputs=["preprocessed_pokemons", "preprocessor"],
                name="preprocess_pokemons",
            ),
            node(
                func=split_data,
                inputs=["preprocessed_pokemons"],
                outputs=[
                    "x_train",
                    "x_val",
                    "x_test",
                    "y_train",
                    "y_val",
                    "y_test",
                ],
                name="split_data",
            ),
        ]
    )
