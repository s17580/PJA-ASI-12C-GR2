import pickle
from sklearn.base import is_classifier
from transform_data import transform_data


# Model release
def release_model():
    with open("pokemon_classifier.pkl", "wb") as f:
        pickle.dump((transform_data, is_classifier), f)
