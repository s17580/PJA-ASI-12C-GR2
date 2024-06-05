from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_pokemons, preprocess_pokemons


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for Pokemon data preparation and preprocessing.

    This pipeline executes the following steps:

    1. prepare_pokemons: Cleans and prepares the raw Pokemon data by handling
                         missing values and removing unnecessary columns.
    2. preprocess_pokemons: Transforms the data for machine learning by scaling
                            numerical features and one-hot encoding categorical
                            features. It also fits a ColumnTransformer.

    Args:
        **kwargs: Additional keyword arguments that can be passed to Kedro nodes.

    Returns:
        Pipeline: A Kedro Pipeline object defining the data preparation and
                  preprocessing steps.
    """
    return pipeline(
        [
            node(
                func=prepare_pokemons,
                inputs="pokemons",
                outputs=["prepared_pokemons", "metadata"],
                name="prepare_pokemons",
            ),
            node(
                func=preprocess_pokemons,
                inputs="prepared_pokemons",
                outputs=["preprocessed_pokemons", "preprocessor"],
                name="preprocess_pokemons",
            ),
        ]
    )
