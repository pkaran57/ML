"""
Name - Karan Patel, PSU ID - 965051876

Class that represents a Perceptron
"""

import logging
import random
import numpy as np


class Perceptron:

    def __init__(self, training_samples, perceptron_class_label):
        self.__training_samples = training_samples
        self.__perceptron_class_label = perceptron_class_label

        self.__num_of_inputs = training_samples[0].features.size + 1
        self.__weights = np.array([random.uniform(-0.5, 0.5) for i in range(self.__num_of_inputs)], dtype=np.double)

        self.__logger = logging.getLogger('Perceptron-target-{}'.format(self.__perceptron_class_label))

    def train_for_an_epoch(self, learning_rate):
        """
        Trains the perceptron using training samples for a single epoch by implementing perceptron learning algorithm and updating weights whenever necessary
        :param learning_rate: Learning rate to use for perceptron learning algorithm
        """

        for sample in self.__training_samples:

            inputs = self.get_inputs(sample.features)

            decision_function_output = Perceptron.decision_function(self.get_net_input(inputs))      # whether perceptron would 'fire' or not
            sample_label_matches_perceptron_class = self.sample_label_equals_perceptron_class(sample)           # ground truth

            if decision_function_output != sample_label_matches_perceptron_class:
                assert (sample_label_matches_perceptron_class - decision_function_output) != 0, 'expected true and predicted class label to not match'
                self.__weights = np.array([weight + (learning_rate * (sample_label_matches_perceptron_class - decision_function_output) * sample_input) for sample_input, weight in zip(inputs, self.__weights)], dtype=np.double)

    def sample_label_equals_perceptron_class(self, sample):
        return 1 if self.__perceptron_class_label == sample.true_class_label else 0

    def get_net_input(self, inputs):
        return np.dot(inputs, self.__weights)

    @staticmethod
    def decision_function(net_input):
        return 1 if net_input > 0 else 0

    def get_inputs(self, features):
        inputs = np.insert(features, 0, 1)
        assert inputs.size == self.__num_of_inputs, 'Expected number of inputs to be {} but got {}'.format(self.__num_of_inputs, inputs.size)
        return inputs
