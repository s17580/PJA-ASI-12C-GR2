"""PJA ASI 12C GR2 file for ensuring the package is executable
as `pja-asi-12c-gr2` and `python -m pja_asi_12c_gr2`
"""
import pandas as pd
import yaml
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

import importlib
from pathlib import Path

from kedro.framework.cli.utils import KedroCliError, load_entry_points
from kedro.framework.project import configure_project
from pja_asi_12c_gr2.pipelines.data_processing.nodes import prepare_pokemons
from pja_asi_12c_gr2.pipelines.split_data_for_training.nodes import split_data
from pja_asi_12c_gr2.pipelines.machine_learning_and_evaluation.nodes import machine_learning,evaluate_model
import wandb


def _find_run_command(package_name):
    try:
        project_cli = importlib.import_module(f"{package_name}.cli")
        # fail gracefully if cli.py does not exist
    except ModuleNotFoundError as exc:
        if f"{package_name}.cli" not in str(exc):
            raise
        plugins = load_entry_points("project")
        run = _find_run_command_in_plugins(plugins) if plugins else None
        if run:
            # use run command from installed plugin if it exists
            return run
        # use run command from the framework project
        from kedro.framework.cli.project import run

        return run
    # fail badly if cli.py exists, but has no `cli` in it
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
    main()

    # df = pd.read_csv('data/02_prepared/prepared_pokemons.csv')

    # data_prepared = prepare_pokemons(df)
    # x_train, x_test, y_train, y_test = split_data(data_prepared)
    
    # # Set up your default hyperparameters
    # with open('./parameters.yaml') as file:
    #     parameters = yaml.load(file, Loader=yaml.FullLoader)
    
    # run = wandb.init(parameters=parameters)
    
    # n_estimators = wandb.parameters.n_estimators
    # max_depth = wandb.parameters.max_depth
    
    # model = machine_learning(x_train, y_train, n_estimators, max_depth)
    # accuracy, roc_auc = evaluate_model(model, x_test, y_test)
    
    # wandb.log({"n_estimators": n_estimators})
    # wandb.log({"max_depth": max_depth})
    # wandb.log({"accuracy": accuracy})
    # # wandb.log({"roc_auc": roc_auc})
    
    # print("n_estimators: ", n_estimators)
    # print("max_depth: ", max_depth)
    # print("accuracy: ", accuracy)
    # # print("roc_auc: ", roc_auc)