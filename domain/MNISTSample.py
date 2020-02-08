"""
Name - Karan Patel, PSU ID - 965051876

Class that represents a sample from the MNIST dataset
"""

import csv
import logging
import random

import numpy as np


class MNISTSample:

    logger = logging.getLogger('MINSTSample')

    def __init__(self, sample_number, true_class_label, features):
        assert type(sample_number) is int, 'Sample number not found!'
        assert 0 <= true_class_label <= 9, 'Expected true class label to be between 0 and 9 for sample #{}'.format(sample_number)
        assert len(features) == 784, 'Expected exactly 784 features for the sample #{}'.format(sample_number)

        self.sample_number = sample_number
        self.true_class_label = int(true_class_label)
        self.inputs = np.array([255] + features, dtype=np.double) / 255
        assert self.inputs.size == 785, 'Expected exactly 785 inputs for sample #{} but got {}'.format(sample_number, self.inputs.size)

    def __str__(self):
        return "Sample #{} : true class label = {}, inputs = {}".format(self.sample_number, self.true_class_label, self.inputs)

    @staticmethod
    def load_and_shuffle_samples_from_dataset(dataset_file_location):
        MNISTSample.logger.info('Loading samples from {} ...'.format(dataset_file_location))

        with open(dataset_file_location) as csv_file:
            samples = [MNISTSample(row_num, int(row[0]), row[1:]) for row_num, row in enumerate(csv.reader(csv_file))]

        random.shuffle(samples)

        MNISTSample.logger.info('Found {} samples in {}'.format(len(samples), dataset_file_location))
        return samples
