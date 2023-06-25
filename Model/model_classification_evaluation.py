import numpy as np
from keras.utils import pad_sequences
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from model.RNN_classification import build_classification_model
from utils.generate_classification_samples import generate_classification_samples, \
    generate_unrealized_classification_samples


def make_predictions(pids, segment_length):
    X_window_1s, X_window_2s, X_window_3s, X_window_4s, X_segments, y = generate_classification_samples(pids)
    max_row_size = max(len(row) for sub_matrix in X_segments for row in sub_matrix)
    X_test_unrealized_bef_reshape_window_1, X_test_unrealized_bef_reshape_window_2, X_test_unrealized_bef_reshape_window_3, X_test_unrealized_bef_reshape_window_4, X_test_unrealized_bef_reshape_segment, y_test_unrealized = generate_unrealized_classification_samples(pids, max_row_size)
    X_test_unrealized_window_1s = np.reshape(X_test_unrealized_bef_reshape_window_1,
                                            (X_test_unrealized_bef_reshape_window_1.shape[0], 20, 3))
    X_test_unrealized_window_2s = np.reshape(X_test_unrealized_bef_reshape_window_2,
                                            (X_test_unrealized_bef_reshape_window_2.shape[0], 40, 3))
    X_test_unrealized_window_3s = np.reshape(X_test_unrealized_bef_reshape_window_3,
                                            (X_test_unrealized_bef_reshape_window_3.shape[0], 60, 3))
    X_test_unrealized_window_4s = np.reshape(X_test_unrealized_bef_reshape_window_4,
                                            (X_test_unrealized_bef_reshape_window_4.shape[0], 80, 3))
    X_test_unrealized_segment = np.reshape(X_test_unrealized_bef_reshape_segment, (X_test_unrealized_bef_reshape_segment.shape[0], X_test_unrealized_bef_reshape_segment.shape[2], 3))

    auc_scores_realized_window_1s, auc_scores_unrealized_window_1s, auc_scores_combination_window_1s = train_and_test(
        X_window_1s, y, X_test_unrealized_window_1s, y_test_unrealized)
    auc_scores_realized_window_2s, auc_scores_unrealized_window_2s, auc_scores_combination_window_2s = train_and_test(
        X_window_2s, y, X_test_unrealized_window_2s, y_test_unrealized)
    auc_scores_realized_window_3s, auc_scores_unrealized_window_3s, auc_scores_combination_window_3s = train_and_test(
        X_window_3s, y, X_test_unrealized_window_3s, y_test_unrealized)
    auc_scores_realized_window_4s, auc_scores_unrealized_window_4s, auc_scores_combination_window_4s = train_and_test(
        X_window_4s, y, X_test_unrealized_window_4s, y_test_unrealized)

    auc_scores_realized_segments, auc_scores_unrealized_segments, auc_scores_combination_segments = train_and_test(X_segments, y, X_test_unrealized_segment, y_test_unrealized)
    # Calculate the average AUC ROC score across all folds
    avg_auc_score_realized_window_1s = sum(auc_scores_realized_window_1s) / len(auc_scores_realized_window_1s)
    avg_auc_score_realized_window_2s = sum(auc_scores_realized_window_2s) / len(auc_scores_realized_window_2s)
    avg_auc_score_realized_window_3s = sum(auc_scores_realized_window_3s) / len(auc_scores_realized_window_3s)
    avg_auc_score_realized_window_4s = sum(auc_scores_realized_window_4s) / len(auc_scores_realized_window_4s)
    # avg_auc_score_unrealized_windows = sum(auc_scores_unrealized_windows) / len(auc_scores_unrealized_windows)
    # avg_auc_score_combination_windows = sum(auc_scores_combination_windows) / len(auc_scores_combination_windows)

    avg_auc_score_realized_segments = sum(auc_scores_realized_segments) / len(auc_scores_realized_segments)
    # avg_auc_score_unrealized_segments = sum(auc_scores_unrealized_segments) / len(auc_scores_unrealized_segments)
    # avg_auc_score_combination_segments = sum(auc_scores_combination_segments) / len(auc_scores_combination_segments)
    print('AUC window_1s', auc_scores_realized_window_1s)
    print('AUC window_2s', auc_scores_realized_window_1s)
    print('AUC window_3s', auc_scores_realized_window_1s)
    print('AUC window_4s', auc_scores_realized_window_1s)
    print('AUC segments', auc_scores_realized_segments)
    # Print the average AUC ROC score
    print('Average AUC ROC score realized intentions (Window 1s):', avg_auc_score_realized_window_1s)
    print('Average AUC ROC score realized intentions (Window 2s):', avg_auc_score_realized_window_2s)
    print('Average AUC ROC score realized intentions (Window 3s):', avg_auc_score_realized_window_3s)
    print('Average AUC ROC score realized intentions (Window 4s):', avg_auc_score_realized_window_4s)
    print('Average AUC ROC score realized intentions (Segments)', avg_auc_score_realized_segments)
    plot_auc_scores(auc_scores_realized_window_1s, auc_scores_realized_window_2s, auc_scores_realized_window_3s, auc_scores_realized_window_4s, auc_scores_realized_segments)


def train_and_test(X, y, X_test_unrealized, y_test_unrealized):
    # Specify the number of folds (example: 5-fold cross-validation)
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
        # Fit the model on the training data
        model = build_classification_model(X_train, y_train)
        # Predict probabilities on the test data
        X_test_realized = np.reshape(X_test, (X_test.shape[0], X_test.shape[2], 3))

        auc_score_realized = test_realized(X_test_realized, y_test_realized, model)
        # auc_score_unrealized = test_unrealized(X_test_unrealized, y_test_unrealized, model)
        # auc_score_combination = test_combination(X_test_realized, y_test_realized, X_test_unrealized, y_test_unrealized, model)
        auc_scores_realized.append(auc_score_realized)
        # auc_scores_unrealized.append(auc_score_unrealized)
        # auc_scores_combination.append(auc_score_combination)

    # plot_auc_scores(auc_scores_realized, auc_scores_realized_dummy, 'Realized')
    # plot_auc_scores(auc_scores_unrealized, auc_scores_unrealized_dummy, 'Unrealized')
    # plot_auc_scores(auc_scores_combination, auc_scores_combination_dummy, 'Combination')
    return auc_scores_realized, auc_scores_unrealized, auc_scores_combination


def test_realized(X_test_realized, y_test_realized, model):
    X_padded = pad_sequences(X_test_realized)
    y_pred_proba_realized = model.predict(X_padded)
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


def plot_auc_scores(auc_scores_window_1s, auc_scores_window_2s, auc_scores_window_3s, auc_scores_window_4s, auc_scores_segments):
    # Combine the arrays into a single list
    data = [auc_scores_window_1s, auc_scores_window_2s, auc_scores_window_3s, auc_scores_window_4s, auc_scores_segments]
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the box and whisker plot
    ax.boxplot(data)
    # Add labels to the x-axis
    ax.set_xticklabels(['1 s', '2 s.', '3 s', '4 s', 'supervised'])
    # Add a title to the plot
    ax.set_title('AUC scores for classification (Box plot)')
    plt.savefig('../results/realized-classification-results-4.png')
    # Display the plot
    plt.show()


if __name__ == '__main__':
    # predict_and_plot_using_auc(pids=[2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35], segment_length=2, epoch=10)
    make_predictions(pids=[2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35], segment_length=2)
