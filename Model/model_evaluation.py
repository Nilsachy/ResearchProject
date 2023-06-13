import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from Model.RNN import build_model
from utils.divide_dataset import divide_dataset
from utils.generate_samples import generate_samples


def predict_and_plot_using_auc(pids, segment_length, epoch):
    aucs = []
    for _ in range(epoch):
        auc = make_predictions(pids, segment_length)
        aucs.append(auc)

    # Sample data for the lines
    x = [i for i in range(1, 11)]
    print(x)

    # Plotting the lines
    plt.plot(x, aucs, color='blue', label='AUC-PR.png positive cases')

    # Adding a legend
    plt.legend()

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('AUC-PR.png score')
    plt.title('Plot of AUC-PR.png score on 10 epochs')
    plt.yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])

    # Display the plot
    plt.show()

def predict_and_plot_using_iou(pids, segment_length, epoch):
    ious_positive_cases = []
    ious_negative_cases = []
    for _ in range(epoch):
        iou_positive_cases, iou_negative_cases = make_predictions(pids, segment_length)
        ious_positive_cases.append(iou_positive_cases)
        ious_negative_cases.append(iou_negative_cases)

    # Sample data for the lines
    x = [i for i in range(1, 11)]
    print(x)

    # Plotting the lines
    plt.plot(x, ious_positive_cases, color='blue', label='IoU - positive cases')
    plt.plot(x, ious_negative_cases, color='red', label='IoU - negative cases')

    # Adding a legend
    plt.legend()

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Plot of IoU metric on 10 epochs (pos and neg cases)')

    # Display the plot
    plt.show()


def make_predictions(pids, segment_length):
    samples = generate_samples(pids, segment_length)
    X_train, X_test, y_train, y_test = divide_dataset(samples)
    model = build_model(X_train, y_train)
    test_data = np.reshape(X_test, (X_test.shape[0], 40, 3))
    predictions = model.predict(test_data)
    filtered_pred, filtered_label = filter_rows_for_non_zero(predictions, y_test)
    auc = evaluate_auc_score_pr(filtered_pred, filtered_label)
    return auc


def iou(predictions, y_test):
    threshold = 0.5
    # Convert predictions to binary values based on the threshold
    binary_predictions = np.where(predictions >= threshold, 1, 0)

    filtered_pred_non_zero, filtered_label = filter_rows_for_non_zero(binary_predictions, y_test)
    iou_positive_cases = evaluate_using_iou(filtered_pred_non_zero, filtered_label)

    filtered_pred_only_zero = filter_rows_for_only_zero(binary_predictions, y_test)
    complement_pred = 1 - filtered_pred_only_zero
    complement_label = np.ones_like(complement_pred)
    iou_negative_cases = evaluate_using_iou(complement_pred, complement_label)

    return iou_positive_cases, iou_negative_cases


def weighted_sum(value1, value2, size1, size2):
    weight1 = size1 / (size1 + size2)
    weight2 = size2 / (size1 + size2)
    weighted_value1 = value1 * weight1
    weighted_value2 = value2 * weight2
    weighted_sum = weighted_value1 + weighted_value2
    return weighted_sum


def filter_rows_for_non_zero(predicted_probs, true_labels):
    non_zero_rows = np.sum(true_labels, axis=1) > 0
    filtered_predicted_probs = predicted_probs[non_zero_rows]
    filtered_true_labels = true_labels[non_zero_rows]
    return filtered_predicted_probs, filtered_true_labels


def filter_rows_for_only_zero(predicted_probs, true_labels):
    zero_rows = np.sum(true_labels, axis=1) == 0
    filtered_predicted_probs = predicted_probs[zero_rows]
    return filtered_predicted_probs


def evaluate_using_iou(predictions, y_test):
    intersection = np.logical_and(predictions, y_test)
    union = np.logical_or(predictions, y_test)
    union_sum = np.sum(union, axis=1)
    iou_scores = np.sum(intersection, axis=1) / union_sum
    average_iou = np.mean(iou_scores)
    return average_iou


#  Other metrics I tried

def my_metric(predicted_probs, y_test):
    predicted_probs_non_zero, true_labels_non_zero = filter_rows_for_non_zero(predicted_probs, y_test)
    average_auc_score_positive_cases = evaluate_auc_score_pr(predicted_probs_non_zero, true_labels_non_zero)

    predicted_probs_only_zero = filter_rows_for_only_zero(predicted_probs, y_test)
    complement_prob = 1 - predicted_probs_only_zero
    complement_label = np.ones_like(complement_prob)
    print(complement_prob)
    print(complement_label)
    average_auc_score_negative_cases = evaluate_auc_score_pr(complement_prob, complement_label)
    return average_auc_score_positive_cases, average_auc_score_negative_cases


def evaluate_auc_score_roc(predicted_probs, true_labels):
    # Calculate PR AUC for each label
    label_aucs_roc = []

    for i in range(len(predicted_probs)):
        # Treat each label as a binary classification problem
        label_preds = predicted_probs[i, :]
        label_true = true_labels[i, :]

        # Calculate PR AUC
        label_auc_roc = roc_auc_score(label_true, label_preds)
        label_aucs_roc.append(label_auc_roc)

    # Average PR AUC values
    average_auc_roc = np.mean(label_aucs_roc)
    print("Average ROC AUC:", average_auc_roc)
    return average_auc_roc


def euclidean_distance(predicted_probs):
    print(predicted_probs)
    zero_array = np.zeros_like(predicted_probs)
    distance = np.linalg.norm(predicted_probs - zero_array)
    print('Euclidean distance:', distance)
    return distance


def evaluate_euclidean_distance(predicted_probs):
    dists = []
    for i in range(len(predicted_probs)):
        label_preds = predicted_probs[i, :]
        euclidean_dist = euclidean_distance(label_preds)
        dists.append(euclidean_dist)
    return np.mean(dists)


def evaluate_using_average_precision(predicted_probs, y_test):
    # Calculate PR AUC for each label
    aucs = []

    for i in range(len(predicted_probs)):
        # Treat each label as a binary classification problem
        label_preds = predicted_probs[i, :]
        label_true = y_test[i, :]
        av_precision_score = calculate_average_precision(label_true, label_preds)
        aucs.append(av_precision_score)

    return np.mean(aucs)


def calculate_average_precision(true_labels, predicted_probs):
    sorted_indices = sorted(range(len(predicted_probs)), key=lambda k: predicted_probs[k], reverse=True)
    sorted_labels = [true_labels[i] for i in sorted_indices]

    precision_values = []
    recall_values = []
    true_positives = 0
    num_positive_labels = sum(true_labels)

    for i in range(len(sorted_labels)):
        true_positives += sorted_labels[i]
        precision = true_positives / (i + 1)
        recall = true_positives / num_positive_labels if num_positive_labels != 0 else 0

        precision_values.append(precision)
        recall_values.append(recall)

    # Calculate the average precision
    ap = sum(precision_values[i] for i in range(len(precision_values)) if recall_values[i] != recall_values[i - 1]) / (
                num_positive_labels or 1)

    return ap


def evaluate_time_to_event_accuracy(predicted_probs, y_test):
    # Calculate PR AUC for each label
    labels_accuracy = []

    for i in range(len(predicted_probs)):
        # Treat each label as a binary classification problem
        label_preds = predicted_probs[i, :]
        label_true = y_test[i, :]
        accuracy = calculate_time_to_event_accuracy(label_true, label_preds)
        labels_accuracy.append(accuracy)
    return np.mean(labels_accuracy)


def calculate_time_to_event_accuracy(ground_truth, predicted):

    # Find the index at which the transition occurs in the ground truth
    ground_truth_index = ground_truth.any(1)

    # Find the index at which the transition occurs in the predicted sequence
    predicted_index = predicted.any(1)

    # Compare the predicted index with the ground truth index
    accuracy = int(ground_truth_index == predicted_index)
    print("Time-to-Event Accuracy: {:.2f}%".format(accuracy * 100))
    return accuracy


def evaluate_auc_score_pr(predicted_probs, y_test):
    # Calculate PR AUC for each label
    label_aucs_pr = []

    for i in range(len(predicted_probs)):
        # Treat each label as a binary classification problem
        label_preds = predicted_probs[i, :]
        label_true = y_test[i, :]

        # Calculate PR AUC
        label_auc_pr = average_precision_score(label_true, label_preds)
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


def evaluate_using_jaccard_similarity(predictions, y_test):
    intersection = np.logical_and(predictions, y_test)
    union = np.logical_or(predictions, y_test)
    union_sum = np.sum(union, axis=1)
    print(intersection)
    print(union)
    print(union_sum)


if __name__ == '__main__':
    predict_and_plot_using_auc(pids=[2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35], segment_length=2, epoch=10)
    # make_predictions(pids=[2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35], segment_length=2)
