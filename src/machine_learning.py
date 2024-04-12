# Machine Learning - Decision Tree Classifier
def machine_learning(x_train, x_test, y_train, y_test, clf):
    clf.fit(x_train, y_train)
    return clf
