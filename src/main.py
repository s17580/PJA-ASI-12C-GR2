from transform_data import transform_data
from machine_learning import machine_learning
from evaluate_model import evaluate_model
from release_model import release_model

# Run modules
x_train, x_test, y_train, y_test, clf = transform_data()
classifier = machine_learning(x_train, x_test, y_train, y_test, clf)
evaluate_model(x_test, y_test, classifier)
release_model()
