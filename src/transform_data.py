from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from load_data import load_data


# Prepare data
def transform_data():
    df = load_data()
    df["Type 2"] = df["Type 2"].fillna("None")  # Filling missing values

    # Features and Labels
    # Dropping "Name" and "Legendary" columns
    X = df.drop(labels=["Name", "Legendary"], axis=1)

    # Target variable
    y = df["Legendary"]

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
