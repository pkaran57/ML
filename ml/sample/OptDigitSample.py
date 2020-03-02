"""
Class that represents a sample from the MNIST dataset
"""

import csv
import logging
import random

import numpy as np


class OptDigitSample:
    """
    Class to help represent a sample in the OptDigit dataset
    """
    logger = logging.getLogger('OptDigitSample')
    expected_features = 64

    def __init__(self, sample_number, true_class_label, features):
        assert type(sample_number) is int, 'Sample number not found!'
        assert 0 <= true_class_label <= 9, 'Expected true class label to be between 0 and 9 for sample #{}'.format(
            sample_number)
        assert len(features) == OptDigitSample.expected_features, 'Expected exactly {} inputs for sample #{}'.format(
            OptDigitSample.expected_features, sample_number)

        self.sample_number = sample_number
        self.true_class_label = int(true_class_label)
        self.features = np.array(features, dtype=np.double)

    def __str__(self):
        return "OptDigit sample #{} : true class label = {}, features = {}".format(self.sample_number,
                                                                                   self.true_class_label,
                                                                                   self.features)

    @staticmethod
    def load_and_shuffle_samples_from_dataset(dataset_file_location, shuffle_samples=True):
        """
        :param scaling_factor features to be scaled by dividing them by scaling_factor
        :param dataset_file_location
        :param shuffle_samples
        :return: list of samples scaled and loaded from dataset_file_location
        """
        OptDigitSample.logger.info('Loading samples from {} ...'.format(dataset_file_location))

        with open(dataset_file_location) as csv_file:
            samples = [OptDigitSample(row_num, int(row[64]), row[:64]) for row_num, row in
                       enumerate(csv.reader(csv_file))]

        if shuffle_samples:
            random.shuffle(samples)

        OptDigitSample.logger.info('Found {} samples in {}'.format(len(samples), dataset_file_location))
        return samples
