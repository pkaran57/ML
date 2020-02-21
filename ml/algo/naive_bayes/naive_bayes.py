# Name - Karan Patel, PSU ID - 965051876

import math
import re
from collections import Counter

import numpy as np

training_file = 'C:\K.E.R Projects\ml\data\yeast_training.txt'
test_file = 'C:\K.E.R Projects\ml\data\yeast_test.txt'

# Useful Lambdas

process_sample_line = lambda sample_line: re.compile(r"\s+").sub(' ', sample_line).strip().split(' ')
convert_to_numpy_array = lambda lst: np.array(lst, dtype=np.double)
str_to_int_converter = lambda str_key: int(str_key)
calculate_probability_density = lambda feature, mean, std: (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(
    -0.5 * (((feature - mean) / std) ** 2))
safe_math_log = lambda num: num if num == 0 else math.log(num)


def get_training_samples_by_label(training_file):
    with open(file=training_file) as training_data_file:
        raw_training_samples = [process_sample_line(line) for line in training_data_file.readlines()]

    num_features = len(raw_training_samples[0]) - 1
    training_data_by_labels = dict()

    for sample in raw_training_samples:
        training_data_by_labels.setdefault(sample[num_features], []).append(sample[:num_features])

    return {key: convert_to_numpy_array(value) for key, value in training_data_by_labels.items()}


def get_test_samples(test_file):
    with open(file=test_file) as test_data_file:
        test_samples = [process_sample_line(line) for line in test_data_file.readlines()]

    num_features = len(test_samples[0]) - 1
    return list(
        map(lambda sample: (sample[num_features], convert_to_numpy_array(sample[:num_features])), test_samples))


def calculate_gaussian(features):
    std = np.std(features)
    return np.mean(features), 0.01 if std < 0.01 else std


def get_highest_probability(probability_densities):
    most_probable_label = None
    highest_probability = -99999999999999

    for label, density in probability_densities.items():
        if density >= highest_probability:
            most_probable_label = label
            highest_probability = density

    return most_probable_label, highest_probability


def get_class_probabilities(samples):
    total_num_of_samples = len(samples)
    counts_by_labels = Counter([true_class_label for true_class_label, feature in samples])
    return {class_label: count / total_num_of_samples for class_label, count in counts_by_labels.items()}


def naive_bayes(training_file, test_file):
    print('Planning phase:\n')

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

    print('\nClassification phase:\n')

    accurate_predictions = 0
    total_predictions = len(test_samples)

    class_probabilities = get_class_probabilities(test_samples)

    for sample_count, (true_class_label, features) in enumerate(test_samples):
        probability_densities = dict()
        for class_label in sorted(list(training_samples_by_label.keys())):
            probability_density = math.log(class_probabilities[class_label])
            for feature, mean, std in zip(features, means[class_label], stds[class_label]):
                probability_density += safe_math_log(calculate_probability_density(feature, mean, std))
            probability_densities[class_label] = probability_density

        predicted_label, probability = get_highest_probability(probability_densities)
        accuracy = 0
        if predicted_label == true_class_label:
            accurate_predictions += 1
            accuracy = 1

        print("ID={0:5d}, predicted={1:3d}, probability = {2:.4f}, true={3:3d}, accuracy={4:4.2f}\n".format(sample_count + 1, int(predicted_label), probability, int(true_class_label), accuracy))

    print("classification accuracy={0:6.4f}".format((accurate_predictions / total_predictions) * 100))


naive_bayes(training_file, test_file)
