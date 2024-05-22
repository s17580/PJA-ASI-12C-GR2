from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_pokemons

def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for Pokemon data processing.

    This pipeline performs the following step:

    1. prepare_pokemons: Cleans and prepares the raw Pokemon data.

    Args:
        **kwargs: Additional keyword arguments that can be passed to Kedro nodes.

    Returns:
        Pipeline: A Kedro Pipeline object defining the entire data processing
                  and splitting workflow.
    """
    return pipeline(
        [
            node(
                func=prepare_pokemons,
                inputs="pokemons",
                outputs=["prepared_pokemons", "prepared_pokemons_columns"],
                name="prepare_pokemons",
            ),
        ]
    )
