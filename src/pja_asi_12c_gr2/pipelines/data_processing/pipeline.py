from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_pokemons


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for Pokemon data preparation.

    This pipeline executes the following steps:

    1. prepare_pokemons: Cleans and prepares the raw Pokemon data by handling
                         missing values and removing unnecessary columns.

    Args:
        **kwargs: Additional keyword arguments that can be passed to Kedro nodes.

    Returns:
        Pipeline: A Kedro Pipeline object defining the data preparation steps.
    """
    return pipeline(
        [
            node(
                func=prepare_pokemons,
                inputs="pokemons",
                outputs=["prepared_pokemons", "metadata"],
                name="prepare_pokemons",
            ),
        ]
    )
