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
from pja_asi_12c_gr2.pipelines.split_data_for_training.nodes import split_data
from pja_asi_12c_gr2.pipelines.machine_learning_and_evaluation.nodes import train_model, evaluate_model, release_model
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
        # Run the pipeline
        main()
        # After running the pipeline, release the model
        release_model(
            evaluation_results=catalog.load("evaluation_results"),
            classifier=catalog.load("classifier"),
            catalog=catalog
        )
