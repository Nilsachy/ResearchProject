import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from model.classification.RNN_classification import build_classification_model
from utils.generate_samples import generate_samples, generate_unrealized_samples


def make_predictions(pids, segment_length):
    X, y = generate_samples(pids, segment_length)
    k_folds = 5
    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True)
    # Perform K-fold cross-validation

    auc_scores_realized = []
    auc_scores_unrealized = []
    auc_scores_combination = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test_realized = y[train_index], y[test_index]

        # Predict probabilities on the test data
        X_test_realized = np.reshape(X_test, (X_test.shape[0], 40, 3))
        X_test_unrealized_bef_reshape, y_test_unrealized = generate_unrealized_samples(pids, segment_length)
        X_test_unrealized = np.reshape(X_test_unrealized_bef_reshape, (X_test_unrealized_bef_reshape.shape[0], 40, 3))

        y_train = convert_to_binary(y_train)
        y_test_realized = convert_to_binary(y_test_realized)
        y_test_unrealized = convert_to_binary(y_test_unrealized)

        # Fit the model on the training data
        model = build_classification_model(X_train, y_train)

        auc_score_realized = test_realized(X_test_realized, y_test_realized, model)
        auc_score_unrealized = test_unrealized(X_test_unrealized, y_test_unrealized, model)
        auc_score_combination = test_combination(X_test_realized, y_test_realized, X_test_unrealized, y_test_unrealized, model)

        auc_scores_realized.append(auc_score_realized)
        auc_scores_unrealized.append(auc_score_unrealized)
        auc_scores_combination.append(auc_score_combination)

    # Calculate the average AUC ROC score across all folds
    avg_auc_score_realized = sum(auc_scores_realized) / len(auc_scores_realized)
    avg_auc_score_unrealized = sum(auc_scores_unrealized) / len(auc_scores_unrealized)
    avg_auc_score_combination = sum(auc_scores_combination) / len(auc_scores_combination)
    print('AUC realized: ', avg_auc_score_realized)
    print('AUC unrealized: ', avg_auc_score_unrealized)
    print('AUC combination: ', avg_auc_score_combination)


def convert_to_binary(y_train):
    y_train_binary = []
    for arr in y_train:
        if np.any(arr == 1):
            y_train_binary.append(1)
        else:
            y_train_binary.append(0)
    return y_train_binary


def test_realized(X_test_realized, y_test_realized, model):
    y_pred_proba_realized = model.predict(X_test_realized)
    auc_score = calculate_auc_score(y_pred_proba_realized, y_test_realized)
    return auc_score


def test_unrealized(X_test_unrealized, y_test_unrealized, model):
    y_pred_proba_unrealized = model.predict(X_test_unrealized)
    auc_score = calculate_auc_score(y_pred_proba_unrealized, y_test_unrealized)
    return auc_score


def test_combination(X_test_realized, y_test_realized, X_test_unrealized, y_test_unrealized, model):
    X_test_combination = np.concatenate((X_test_realized, X_test_unrealized), axis=0)
    y_test_combination = np.concatenate((y_test_realized, y_test_unrealized), axis=0)
    y_pred_proba_combination = model.predict(X_test_combination)
    auc_score = calculate_auc_score(y_pred_proba_combination, y_test_combination)
    return auc_score


def calculate_auc_score(y_pred_proba, y_test):
    # Calculate the AUC ROC score for the fold
    flattened_pred = y_pred_proba.flatten()
    flattened_y_test = y_test.flatten()
    auc_score = roc_auc_score(flattened_y_test, flattened_pred)
    return auc_score


def plot_auc_scores(auc_scores, auc_scores_dummy, configuration):
    # Combine the arrays into a single list
    data = [auc_scores, auc_scores_dummy]
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the box and whisker plot
    ax.boxplot(data)
    # Add labels to the x-axis
    ax.set_xticklabels(['Trained model', 'Baseline model'])
    # Add a title to the plot
    ax.set_title('AUC scores (Box plot) - ' + configuration)
    plt.savefig('../results/' + configuration + '-results.png')
    # Display the plot
    plt.show()


if __name__ == '__main__':
    make_predictions(pids=[2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35], segment_length=2)