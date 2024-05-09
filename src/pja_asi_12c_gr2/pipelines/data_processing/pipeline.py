from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_pokemons, preprocess_pokemons, split_data


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the Pokemon data processing and splitting pipeline.

    Returns:
        Pipeline: A Kedro pipeline for preprocessing and splitting the Pokemon dataset.
    """
    return pipeline(
        [
            node(
                func=prepare_pokemons,
                inputs="pokemons",
                outputs=["prepared_pokemons"],
                name="prepare_pokemons",
            ),
            node(
                func=preprocess_pokemons,
                inputs="prepared_pokemons",
                outputs=["preprocessor"],
                name="preprocess_pokemons",
            ),
            node(
                func=split_data,
                inputs=["prepared_pokemons", "params:split_data"],
                outputs=["x_train", "x_val", "x_test", "y_train", "y_val", "y_test"],
                name="split_data",
            ),
        ]
    )
