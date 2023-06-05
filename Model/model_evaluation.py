import numpy as np

from Model.RNN import build_model
from utils.divide_dataset import divide_dataset
from utils.generate_samples import generate_samples


def make_predictions(pids, segment_length):
    samples = generate_samples(pids, segment_length)
    X_train, X_test, y_train, y_test = divide_dataset(samples)
    model = build_model(X_train, y_train)
    test_data = np.reshape(X_test, (X_test.shape[0], 200, 3))
    predictions = model.predict(test_data)
    # Set the threshold value
    threshold = 0.5
    # Convert predictions to binary values based on the threshold
    binary_predictions = np.where(predictions >= threshold, 1, 0)
    # evaluate_using_iou(binary_predictions, y_test)
    cla = evaluate_using_cla(binary_predictions, y_test)
    print(cla)


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
