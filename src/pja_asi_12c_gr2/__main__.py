import pandas as pd
import yaml
import numpy as np
import importlib
from pathlib import Path
from kedro.framework.cli.utils import KedroCliError, load_entry_points
from kedro.framework.project import configure_project
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hooks_manager
from kedro.framework.session import KedroSession
from kedro.io import DataCatalog
from pja_asi_12c_gr2.pipelines.data_processing.nodes import prepare_pokemons
from pja_asi_12c_gr2.pipelines.data_science import split_data, train_model, evaluate_model
from pja_asi_12c_gr2.pipelines.deployment.nodes import select_best_model, release_model
import wandb

def _find_run_command(package_name):
    try:
        project_cli = importlib.import_module(f"{package_name}.cli")
    except ModuleNotFoundError as exc:
        if f"{package_name}.cli" not in str(exc):
            raise
        plugins = load_entry_points("project")
        run = _find_run_command_in_plugins(plugins) if plugins else None
        if run:
            return run
        from kedro.framework.cli.project import run
        return run
    if not hasattr(project_cli, "cli"):
        raise KedroCliError(f"Cannot load commands from {package_name}.cli")
    return project_cli.run

def _find_run_command_in_plugins(plugins):
    for group in plugins:
        if "run" in group.commands:
            return group.commands["run"]

def main(*args, **kwargs):
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = _find_run_command(package_name)
    run(*args, **kwargs)

if __name__ == "__main__":
    with KedroSession.create(package_name="pja_asi_12c_gr2") as session:
        context = session.load_context()
        catalog = context.catalog
        params = context.params

        if params.get("select_best_model", False):
            # Run the pipeline including the select_best_model node
            main(pipeline_name="__default__")
        else:
            # Run only the data_science pipeline
            main(pipeline_name="data_science")

        # After running the pipeline, release the model
        classifier = catalog.load("classifier")
        release_model(classifier)
