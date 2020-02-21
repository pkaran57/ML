# Name - Karan Patel, PSU ID - 965051876

import math
import re
from collections import Counter

import numpy as np

training_file = 'C:\K.E.R Projects\ml\data\yeast_training.txt'
test_file = 'C:\K.E.R Projects\ml\data\yeast_test.txt'

# Useful Lambdas

process_sample_line = lambda sample_line: re.compile(r"\s+").sub(' ', sample_line).strip().split(' ')           # return a list of attributes and true class label (last item in list) for a sample given a line from the data file
convert_to_numpy_array = lambda lst: np.array(lst, dtype=np.double)                                             # converts a list into a np array of doubles
str_to_int_converter = lambda str_key: int(str_key)                                                             # converts a string to int
calculate_probability_density = lambda feature, mean, std: (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(
    -0.5 * (((feature - mean) / std) ** 2))                                                                # probability density function (PDF)
safe_math_log = lambda num: num if num == 0 else math.log(num)                                                  # prevents executing natural log function on 0 (which is undefined)


def get_training_samples_by_label(training_file):
    """
    Read and return training samples from training_file
    :param training_file: path to file containing training data
    :return: dictionary of class label to all samples belonging to that class
    """
    with open(file=training_file) as training_data_file:
        raw_training_samples = [process_sample_line(line) for line in training_data_file.readlines()]

    num_features = len(raw_training_samples[0]) - 1
    training_data_by_labels = dict()

    for sample in raw_training_samples:
        training_data_by_labels.setdefault(sample[num_features], []).append(sample[:num_features])

    return {key: convert_to_numpy_array(value) for key, value in training_data_by_labels.items()}


def get_test_samples(test_file):
    """
    Read and return test samples from test_file
    :param test_file: path to file containing test data
    :return: list of tuples. First item in tuple is true class label and the second item is a list of features belonging to a test sample
    """
    with open(file=test_file) as test_data_file:
        test_samples = [process_sample_line(line) for line in test_data_file.readlines()]

    num_features = len(test_samples[0]) - 1
    return list(
        map(lambda sample: (sample[num_features], convert_to_numpy_array(sample[:num_features])), test_samples))


def calculate_gaussian(features):
    """
    Calculate mean and standard deviation for a given set of features
    :return: tuple with first item being mean and the second item being standard deviation
    """
    std = np.std(features)
    return np.mean(features), 0.01 if std < 0.01 else std


def get_prediction_and_accuracy(probability_densities, true_class_label):
    """
    Compute prediction, probability of the prediction and accuracy for the prediction
    :param probability_densities: probability densities for each class label for a given sample
    :param true_class_label: true class label of a given sample
    :return: A tuple of prediction, probability of the prediction and accuracy for the prediction
    """
    most_probable_label = None
    highest_probability = -99999999999999

    for label, density in probability_densities.items():
        if density >= highest_probability:
            most_probable_label = label
            highest_probability = density

    accuracy = 0
    if most_probable_label == true_class_label:
        accuracy = 1 / list(probability_densities.values()).count(highest_probability)

    return most_probable_label, highest_probability, accuracy


def get_class_probabilities(samples):
    """
    :return: dictionary of class label to probability for that class label in samples
    """
    total_num_of_samples = len(samples)
    counts_by_labels = Counter([true_class_label for true_class_label, feature in samples])
    return {class_label: count / total_num_of_samples for class_label, count in counts_by_labels.items()}


def naive_bayes(training_file, test_file):
    """
    Top level function
    """

    # Training phase

    training_samples_by_label = get_training_samples_by_label(training_file)
    test_samples = get_test_samples(test_file)

    means = {label: [] for label in training_samples_by_label.keys()}
    stds = {label: [] for label in training_samples_by_label.keys()}

    for predicted_label in sorted(list(training_samples_by_label.keys()), key=str_to_int_converter):
        total_num_of_features = len(training_samples_by_label[predicted_label][0])
        for i in range(total_num_of_features):
            mean, std = calculate_gaussian(training_samples_by_label[predicted_label][:, i])
            means[predicted_label].append(mean)
            stds[predicted_label].append(std)

            print("Class {0:d}, attribute {1:d}, mean = {2:.2f}, std = {3:.2f}".format(int(predicted_label), i + 1, mean, std))

    # Classification phase

    accuracy = 0
    total_predictions = len(test_samples)

    class_probabilities = get_class_probabilities(test_samples)

    for sample_count, (true_class_label, features) in enumerate(test_samples):
        probability_densities = dict()
        for class_label in sorted(list(training_samples_by_label.keys())):
            probability_density = math.log(class_probabilities[class_label])
            for feature, mean, std in zip(features, means[class_label], stds[class_label]):
                probability_density += safe_math_log(calculate_probability_density(feature, mean, std))
            probability_densities[class_label] = probability_density

        predicted_label, probability, accuracy_for_sample = get_prediction_and_accuracy(probability_densities, true_class_label)
        accuracy += accuracy_for_sample


        print("ID={0:5d}, predicted={1:3d}, probability = {2:.4f}, true={3:3d}, accuracy={4:4.2f}\n".format(sample_count + 1, int(predicted_label), probability, int(true_class_label), accuracy_for_sample))

    print("classification accuracy={0:6.4f}".format((accuracy / total_predictions) * 100))


naive_bayes(training_file, test_file)
