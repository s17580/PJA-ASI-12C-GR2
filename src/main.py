import pandas as pd
import numpy as np


# LoadData Module
def loadData():
    dataset = pd.read_csv("pokemon.csv", index_col=False)
    return dataset


# Run modules
loadData()
