from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_data():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
    )

    return X_train, X_test, y_train, y_test
