import numpy as np
from sklearn.model_selection import train_test_split


def divide_dataset(samples):
    X, y = divide_X_train_and_y_train(samples)
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
    return train_data, test_data, train_labels, test_labels


def divide_X_train_and_y_train(samples):
    # Extract the first positions into a new array
    X = np.array([s[0] for s in samples])
    # Extract the second positions into a new array
    y = np.array([s[1] for s in samples])
    return X, y
