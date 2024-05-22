from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_pokemons

def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for Pokemon data processing.

    This pipeline performs the following step:

    1. preprocess_pokemons:  Preprocesses the data for machine learning by
                            scaling numerical features and one-hot encoding
                            categorical features.

    Args:
        **kwargs: Additional keyword arguments that can be passed to Kedro nodes.

    Returns:
        Pipeline: A Kedro Pipeline object defining the entire data processing
                  and splitting workflow.
    """
    return pipeline(
        [
            node(
                func=preprocess_pokemons,
                inputs="prepared_pokemons",
                outputs=["preprocessed_pokemons", "preprocessor"],
                name="preprocess_pokemons",
            ),
        ]
    )
