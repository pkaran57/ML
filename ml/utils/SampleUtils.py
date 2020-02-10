from random import shuffle


def get_training_samples_by_target_labels(training_samples):
    training_samples_by_target_labels = dict()
    for sample in training_samples:
        if sample.true_class_label not in training_samples_by_target_labels:
            training_samples_by_target_labels[sample.true_class_label] = list()
        training_samples_by_target_labels[sample.true_class_label].append(sample)
    return training_samples_by_target_labels


def get_fraction_of_equally_distributed_samples(num_target_labels, num_samples, training_sample_fraction, training_samples_by_target_labels):
    num_training_samples_of_each_type_in_subset = int((num_samples * training_sample_fraction) / num_target_labels)
    training_samples_subset = []
    for samples_of_a_type in training_samples_by_target_labels.values():
        training_samples_subset.extend(samples_of_a_type[:num_training_samples_of_each_type_in_subset])
    shuffle(training_samples_subset)
    return training_samples_subset