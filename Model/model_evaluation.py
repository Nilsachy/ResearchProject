import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from Model.RNN import build_model
from utils.divide_dataset import divide_dataset
from utils.generate_samples import generate_samples


def make_predictions(pids, segment_length):
    X, y = generate_samples(pids, segment_length)
    # Specify the number of folds (example: 5-fold cross-validation)
    k_folds = 5
    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True)
    # Perform K-fold cross-validation
    auc_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit the model on the training data
        model = build_model(X_train, y_train)
        # Predict probabilities on the test data
        test_data = np.reshape(X_test, (X_test.shape[0], 40, 3))
        y_pred_proba = model.predict(test_data)
        # Calculate the AUC ROC score for the fold
        flattened_pred = y_pred_proba.flatten()
        flattened_y_test = y_test.flatten()
        auc_score = roc_auc_score(flattened_y_test, flattened_pred)
        auc_scores.append(auc_score)
    # Calculate the average AUC ROC score across all folds
    avg_auc_score = sum(auc_scores) / len(auc_scores)
    plot_auc_scores(auc_scores)
    # Print the average AUC ROC score
    print("Average AUC ROC score:", avg_auc_score)


def plot_auc_scores(auc_scores):
    # Sample data for the lines
    x = [i for i in range(1, 6)]
    print(x)

    # Plotting the lines
    plt.plot(x, auc_scores, color='blue', label='AUC-ROC')

    # Adding a legend
    plt.legend()

    # Adding labels and title
    plt.xlabel('Fold')
    plt.ylabel('AUC-ROC score')
    plt.title('Plot of AUC-ROC score on 5-fold cross validation')
    plt.xticks([1, 2, 3, 4, 5])
    plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # predict_and_plot_using_auc(pids=[2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35], segment_length=2, epoch=10)
    make_predictions(pids=[2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35], segment_length=2)
