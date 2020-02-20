import math
import re

import numpy as np

training_file = 'C:\K.E.R Projects\ml\data\yeast_training.txt'
test_file = 'C:\K.E.R Projects\ml\data\yeast_test.txt'

regex = re.compile(r"\s+")
process_sample_line = lambda sample_line: regex.sub(' ', sample_line).strip().split(' ')
convert_to_numpy_array = lambda lst: np.array(lst, dtype=np.double)


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


calculate_probability_density = lambda feature, mean, std: (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-0.5 * (((feature - mean) / std) ** 2))

safe_math_log = lambda num: num if num == 0 else math.log(num)

def naive_bayes(training_file, test_file):
    training_samples_by_label = get_training_samples_by_label(training_file)
    test_samples = get_test_samples(test_file)

    means = {label: [] for label in training_samples_by_label.keys()}
    stds = {label: [] for label in training_samples_by_label.keys()}

    for label, features in training_samples_by_label.items():
        for i in range(len(features[0])):
            mean, std = calculate_gaussian(training_samples_by_label[label][:,i])
            means[label].append(mean)
            stds[label].append(std)

    accurate_predictions = 0
    total_predictions = len(test_samples)

    for true_class_label, features in test_samples:
        probability_densities = dict()
        for class_label in sorted(list(training_samples_by_label.keys())):
            probability_density = 0
            for feature, mean, std in zip(features, means[class_label], stds[class_label]):
                probability_density += safe_math_log(calculate_probability_density(feature, mean, std))
            probability_densities[class_label] = probability_density

        probability_densities_arr = convert_to_numpy_array(list(probability_densities.values()))        # fix this
        ind = np.argmax(probability_densities_arr)
        if ind + 1 == int(true_class_label):
            accurate_predictions += 1

    print(accurate_predictions / total_predictions)




naive_bayes(training_file, test_file)
