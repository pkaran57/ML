import logging
import math
import random

import numpy as np


class SingleLaterNeuralNet:

    def __init__(self, num_hidden_units, training_samples, validation_samples, num_target_labels, **kwargs):
        self.__num_hidden_units = num_hidden_units
        self.__training_samples = training_samples
        self.__validation_samples = validation_samples
        self.__num_target_labels = num_target_labels

        self.__input_to_hidden_weights = kwargs.get('input_to_hidden_weights',
                                                    SingleLaterNeuralNet.initialize_weight_matrix(shape=(num_hidden_units, training_samples[0].inputs.size)))

        self.__hidden_to_output_weights = kwargs.get('hidden_to_output_weights',
                                                     SingleLaterNeuralNet.initialize_weight_matrix(shape=(num_target_labels, num_hidden_units + 1)))

        self.__input_to_hidden_weights_delta_from_previous_itr = self.get_zero_matrix(self.__input_to_hidden_weights.shape)
        self.__hidden_to_output_weights_delta_from_previous_itr = self.get_zero_matrix(self.__hidden_to_output_weights.shape)

        self.__logger = logging.getLogger('NeuralNet:')

    def train(self, num_of_epochs, learning_rate=0.1, momentum=0):
        for _ in range(num_of_epochs):
            self.__logger.info('Starting epoch #{}'.format(_))
            self.__train_for_an_epoch(learning_rate, momentum)

    def __train_for_an_epoch(self, learning_rate=0.1, momentum=0):
        samples_processed = 0
        for sample in self.__training_samples:

            hidden_activations = self.get_hidden_activations(sample)
            output_activations = self.get_output_activations(hidden_activations)

            output_error_terms = self.get_output_error_terms(output_activations, sample)
            hidden_error_terms = self.get_hidden_error_terms(hidden_activations, output_error_terms)

            hidden_to_output_delta = self.calculate_hidden_to_output_delta(hidden_activations, learning_rate, momentum, output_error_terms)
            self.__hidden_to_output_weights = self.__hidden_to_output_weights + hidden_to_output_delta

            input_to_hidden_delta = self.calculate_input_to_hidden_delta(hidden_error_terms, learning_rate, momentum, sample)
            self.__input_to_hidden_weights = self.__input_to_hidden_weights + input_to_hidden_delta

            self.__hidden_to_output_weights_delta_from_previous_itr = hidden_to_output_delta
            self.__input_to_hidden_weights_delta_from_previous_itr = input_to_hidden_delta

            samples_processed += 1
            self.__logger.info('Processed {} samples'.format(samples_processed))

    def calculate_input_to_hidden_delta(self, hidden_error_terms, learning_rate, momentum, sample):
        input_to_hidden_delta = []
        for i_num, i in enumerate(hidden_error_terms):
            delta = []
            for j_num, j in enumerate(sample.inputs):
                delta.append((learning_rate * i * j) + (
                            momentum * self.__input_to_hidden_weights_delta_from_previous_itr[i_num, j_num]))
            input_to_hidden_delta.append(delta)
        return np.array(input_to_hidden_delta, dtype=np.double)

    def calculate_hidden_to_output_delta(self, hidden_activations, learning_rate, momentum, output_error_terms):
        hidden_to_output_delta = []
        for i_num, i in enumerate(output_error_terms):
            delta = []
            for j_num, j in enumerate([1.0] + hidden_activations):
                delta.append((learning_rate * i * j) + (
                            momentum * self.__hidden_to_output_weights_delta_from_previous_itr[i_num, j_num]))
            hidden_to_output_delta.append(delta)
        return np.array(hidden_to_output_delta, dtype=np.double)

    @staticmethod
    def get_output_error_terms(output_activations, sample):
        output_error_terms = [
            activation * (1 - activation) * (0.9 if act_num == sample.true_class_label else 0.1 - activation) for
            act_num, activation in enumerate(output_activations)]
        return output_error_terms

    def get_output_activations(self, hidden_activations):
        output_activations = list(map(SingleLaterNeuralNet.sigmoid_activation_function,
                                      np.dot(self.__hidden_to_output_weights, [1.0] + hidden_activations)))
        assert len(output_activations) == self.__num_target_labels, 'Unexpected output activation array size'
        return output_activations

    def get_hidden_activations(self, sample):
        hidden_activations = list(map(SingleLaterNeuralNet.sigmoid_activation_function,
                                      np.dot(self.__input_to_hidden_weights, sample.inputs)))
        assert len(hidden_activations) == self.__num_hidden_units, 'Unexpected hidden activation array size'
        return hidden_activations

    def get_hidden_error_terms(self, hidden_activations, output_error_terms):
        hidden_error_terms = []
        for num, hidden_activation in enumerate(hidden_activations):
            error_term = hidden_activation * (1 - hidden_activation) * (
                np.dot(output_error_terms, self.__hidden_to_output_weights[:, num + 1]))
            hidden_error_terms.append(error_term)
        return hidden_error_terms

    @staticmethod
    def sigmoid_activation_function(net_input):
        return 1 / (1 + math.exp(-net_input))

    @staticmethod
    def initialize_weight_matrix(shape):
        weights = [[random.uniform(-0.5, 0.5) for i in range(shape[1])] for _ in range(shape[0])]
        return np.array(weights, dtype=np.double)

    @staticmethod
    def get_zero_matrix(shape):
        return np.zeros(shape=shape, dtype=np.double)
