import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def load_nist():
    rng = np.random.RandomState(42)
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False, random_state=rng)
    return X_train, y_train, X_test, y_test


def load_iris(return_train_test_split=True):
    rng = np.random.RandomState(42)
    data = np.load("./data/flowers.npz", allow_pickle=True)
    X, y = data['X'], data['y']
    if return_train_test_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False, random_state=rng)
    else:
        X_train, y_train, X_test, y_test = X, y, np.array([]), np.array([])
    return X_train, y_train, X_test, y_test
