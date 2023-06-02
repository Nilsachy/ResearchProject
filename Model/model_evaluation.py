import numpy as np

from Model.CNN import build_model
from utils.generate_samples import generate_samples


def make_predictions(pids, segment_length):
    samples = generate_samples(pids, segment_length)
    X_train, y_train = divide_X_train_and_y_train(samples)
    model = build_model(X_train, y_train)
    print(model)


def divide_X_train_and_y_train(samples):
    # Extract the first positions into a new array
    X_train = np.array([s[0] for s in samples])
    # Extract the second positions into a new array
    y_train = np.array([s[1] for s in samples])
    return X_train, y_train


if __name__ == '__main__':
    make_predictions(pids=[2, 3], segment_length=2)
