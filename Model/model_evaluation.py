import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from Model.RNN import build_model
from utils.divide_dataset import divide_dataset
from utils.generate_samples import generate_samples


def make_predictions(pids, segment_length):
    samples = generate_samples(pids, segment_length)
    X_train, X_test, y_train, y_test = divide_dataset(samples)
    model = build_model(X_train, y_train)
    test_data = np.reshape(X_test, (X_test.shape[0], 40, 3))
    predictions = model.predict(test_data)
    evaluate_auc_score_pr(predictions, y_test)
    # Set the threshold value
    # threshold = 0.5
    # Convert predictions to binary values based on the threshold
    # binary_predictions = np.where(predictions >= threshold, 1, 0)
    # evaluate_using_iou(binary_predictions, y_test)
    # cla = evaluate_using_cla(binary_predictions, y_test)
    # print(cla)


def evaluate_auc_score_pr(predicted_probs, y_test):
    # Calculate PR AUC for each label
    label_aucs_pr = []

    for i in range(len(predicted_probs)):
        # Treat each label as a binary classification problem
        label_preds = predicted_probs[i, :]
        label_true = y_test[i, :]

        # Calculate PR AUC
        label_auc_pr = average_precision_score(label_true, label_preds)
        if np.isnan(label_auc_pr):
            label_auc_pr = 0.0
        label_aucs_pr.append(label_auc_pr)

    # Average PR AUC values
    average_auc_pr = np.mean(label_aucs_pr)
    print("Average PR AUC:", average_auc_pr)
    return average_auc_pr


def evaluate_mean_auc_score(predicted_probs, y_test):
    aucs = []
    for i in range(len(predicted_probs)):
        label_preds = predicted_probs[i, :]
        label_true = y_test[i, :]
        auc = calculate_auc(label_true, label_preds)
        aucs.append(auc)
    print(aucs)
    return np.mean(aucs)


def calculate_auc(true_labels, predicted_probabilities):
    # Combine true labels and predicted probabilities into a single list
    labels_and_probs = list(zip(true_labels, predicted_probabilities))

    # Sort the combined list by predicted probabilities in descending order
    labels_and_probs = sorted(labels_and_probs, key=lambda x: x[1], reverse=True)

    # Initialize variables to keep track of true positive (TP) and false positive (FP) counts
    tp = 0
    fp = 0

    # Initialize variables to keep track of previous true positive count and AUC
    prev_tp = 0
    auc = 0

    # Iterate through the sorted list
    for label, prob in labels_and_probs:
        # Update the counts
        if label == 1:
            tp += 1
        else:
            fp += 1

        # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
        if sum(true_labels) > 0:
            tpr = tp / sum(true_labels)
        else:
            tpr = 0.0

        if (len(true_labels) - sum(true_labels)) > 0:
            fpr = fp / (len(true_labels) - sum(true_labels))
        else:
            fpr = 0.0

        # Update the AUC by summing the area under the curve
        auc += (tpr - prev_tp) * (fpr + (fpr - (fp - 1) / (len(true_labels) - sum(true_labels)))) / 2

        # Update the previous true positive count
        prev_tp = tpr

    return auc


def evaluate_using_cla(predictions, y_test):
    total_characters = 0
    correct_characters = 0

    for i in range(len(predictions)):
        prediction = predictions[i]
        true_label = y_test[i]

        for j in range(len(prediction)):
            total_characters += 1
            if prediction[j] == true_label[j]:
                correct_characters += 1

    character_accuracy = correct_characters / total_characters

    return character_accuracy


def evaluate_using_iou(predictions, y_test):
    intersection = np.logical_and(predictions, y_test)
    union = np.logical_or(predictions, y_test)
    union_sum = np.sum(union, axis=1)
    union_sum[union_sum == 0] = 1
    iou_scores = np.sum(intersection, axis=1) / union_sum
    average_iou = np.mean(iou_scores)
    print(average_iou)


if __name__ == '__main__':
    make_predictions(pids=[2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35], segment_length=2)
