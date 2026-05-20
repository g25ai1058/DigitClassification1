from utils import *

X_train, X_test, y_train, y_test = load_digits_dataset_files()
predicted, clf = train_test_model(X_train, X_test, y_train, y_test)
performance_model(X_test, y_test, predicted, clf)