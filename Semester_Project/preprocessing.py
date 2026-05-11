import numpy as np

def normalize_data(X):
    return X / 255.0

def reshape_for_nn(X):
    return X.reshape(-1, 28, 28, 1)

def preprocess_all(X_train, X_test):
    X_train = normalize_data(X_train)
    X_test  = normalize_data(X_test)
    return X_train, X_test