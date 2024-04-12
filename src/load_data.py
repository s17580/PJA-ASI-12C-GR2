import pandas as pd


# Load data
def load_data():
    df = pd.read_csv("data\\pokemon.csv", index_col=False)
    return df
