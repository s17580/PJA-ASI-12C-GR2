from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for Pokemon data processing and splitting.

    This pipeline performs the following step:

    1. split_data: Splits the preprocessed data into training, validation, and
                   testing sets.

    Args:
        **kwargs: Additional keyword arguments that can be passed to Kedro nodes.

    Returns:
        Pipeline: A Kedro Pipeline object defining the entire data processing
                  and splitting workflow.
    """
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_pokemons", "params:split_data"],
                outputs=["x_train", "x_val", "x_test", "y_train", "y_val", "y_test"],
                name="split_data",
            ),
        ]
    )
