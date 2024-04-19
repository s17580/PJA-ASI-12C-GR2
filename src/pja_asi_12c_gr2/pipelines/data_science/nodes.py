from kedro.io import DataCatalog
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def transform_data(pokemons):

    pokemons["Type 2"] = pokemons["Type 2"].fillna("None")  # Filling missing values

    # Features and Labels
    # Dropping "Name" and "Legendary" columns
    X = pokemons.drop(labels=["Name", "Legendary"], axis=1)

    # Target variable
    y = pokemons["Legendary"]

    # Encoding categorical features "Type 1" and "Type 2"
    categorical_features = ["Type 1", "Type 2"]
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

    # Creating a column transformer to transform categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    # Creating a pipeline to transform data before applying the classifier
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=0)),
        ]
    )

    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    return x_train, x_test, y_train, y_test, clf


def machine_learning(x_train, x_test, y_train, y_test, clf):
    clf.fit(x_train, y_train)
    return clf


def evaluate_model(x_test, y_test, classifier):
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    conf_matrix = confusion_matrix(y_test, y_pred)

    evaluation_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),  # konwersja na listę dla obsługi JSON
    }

    return evaluation_results


def release_model(catalog: DataCatalog, evaluation_results: dict, classifier):
    catalog.save("evaluation_results", evaluation_results)
