import logging
import math
import random

import numpy as np

from ml.utils.ArrayUtils import convert_to_np_array


class SingleLaterNeuralNet:

    def __init__(self, num_hidden_units, training_samples, validation_samples, num_target_labels, **kwargs):
        self._num_hidden_units = num_hidden_units
        self._training_samples = training_samples
        self._validation_samples = validation_samples
        self._num_target_labels = num_target_labels

        self._input_to_hidden_weights = kwargs.get('input_to_hidden_weights',
                                                    SingleLaterNeuralNet.initialize_weight_matrix(shape=(num_hidden_units, training_samples[0].inputs.size)))

        self._hidden_to_output_weights = kwargs.get('hidden_to_output_weights',
                                                     SingleLaterNeuralNet.initialize_weight_matrix(shape=(num_target_labels, num_hidden_units + 1)))

        self._input_to_hidden_weights_delta_from_previous_itr = self.get_zero_matrix(self._input_to_hidden_weights.shape)
        self._hidden_to_output_weights_delta_from_previous_itr = self.get_zero_matrix(self._hidden_to_output_weights.shape)

        self._training_accuracies_per_epoch = []
        self._validation_accuracies_per_epoch = []

        self._logger = logging.getLogger('NeuralNet:')

    def train(self, num_of_epochs, learning_rate=0.1, momentum=0):
        self.compute_accuracy(0)

        for epoch_num in range(1, num_of_epochs + 1):
            self._logger.info('Starting training for epoch #{}'.format(epoch_num))
            self._train_for_an_epoch(learning_rate, momentum)

            self.compute_accuracy(epoch_num)

    def compute_accuracy(self, epoch_num):
        for samples in self._training_samples, self._validation_samples:
            correct_predictions = 0
            total_num_predictions = len(samples)

            for sample in samples:

                hidden_activations = self._get_hidden_activations(sample)
                output_activations = self._get_output_activations(hidden_activations)

                if output_activations.argmax() == sample.true_class_label:
                    correct_predictions += 1

            accuracy = (correct_predictions / total_num_predictions) * 100

            is_validation_sample = True if samples is self._validation_samples else False

            if is_validation_sample:
                self._validation_accuracies_per_epoch.append(accuracy)
            else:
                self._training_accuracies_per_epoch.append(accuracy)

            self._logger.info("For epoch #{}, accuracy for {} dataset = {}".format(epoch_num, 'validation samples' if is_validation_sample else 'training samples', accuracy))

    def _train_for_an_epoch(self, learning_rate=0.1, momentum=0):
        for sample in self._training_samples:

            hidden_activations = self._get_hidden_activations(sample)
            output_activations = self._get_output_activations(hidden_activations)

            output_error_terms = self.get_output_error_terms(output_activations, sample)
            hidden_error_terms = self._get_hidden_error_terms(hidden_activations, output_error_terms)

            hidden_to_output_delta = self._calculate_hidden_to_output_delta(hidden_activations, learning_rate, momentum, output_error_terms, self._hidden_to_output_weights_delta_from_previous_itr)
            self._hidden_to_output_weights = self._hidden_to_output_weights + hidden_to_output_delta

            input_to_hidden_delta = self._calculate_input_to_hidden_delta(hidden_error_terms, learning_rate, momentum, sample)
            self._input_to_hidden_weights = self._input_to_hidden_weights + input_to_hidden_delta

            self._hidden_to_output_weights_delta_from_previous_itr = hidden_to_output_delta
            self._input_to_hidden_weights_delta_from_previous_itr = input_to_hidden_delta

    def _calculate_input_to_hidden_delta(self, hidden_error_terms, learning_rate, momentum, sample):
        momentum_matrix = self._input_to_hidden_weights_delta_from_previous_itr * momentum
        error_times_learning_rate = hidden_error_terms * learning_rate

        return (sample.inputs[np.newaxis] * error_times_learning_rate[np.newaxis].T) + momentum_matrix

    @staticmethod
    def _calculate_hidden_to_output_delta(hidden_activations, learning_rate, momentum, output_error_terms, hidden_to_output_weights_delta_from_previous_itr):
        momentum_matrix = hidden_to_output_weights_delta_from_previous_itr * momentum
        hidden_activations_with_bias = np.insert(hidden_activations, 0, 1)
        error_times_learning_rate = output_error_terms * learning_rate

        return (hidden_activations_with_bias[np.newaxis] * error_times_learning_rate[np.newaxis].T) + momentum_matrix

    @staticmethod
    def get_output_error_terms(output_activations, sample):
        output_error_terms = [
            activation * (1 - activation) * ((0.9 if act_num == sample.true_class_label else 0.1) - activation) for
            act_num, activation in enumerate(output_activations)]
        return convert_to_np_array(output_error_terms)

    def _get_output_activations(self, hidden_activations):
        output_activations = list(map(SingleLaterNeuralNet.sigmoid_activation_function,
                                      np.dot(self._hidden_to_output_weights, np.insert(hidden_activations, 0, 1))))
        assert len(output_activations) == self._num_target_labels, 'Unexpected output activation array size'
        return convert_to_np_array(output_activations)

    def _get_hidden_activations(self, sample):
        hidden_activations = list(map(SingleLaterNeuralNet.sigmoid_activation_function,
                                      np.dot(self._input_to_hidden_weights, sample.inputs)))
        assert len(hidden_activations) == self._num_hidden_units, 'Unexpected hidden activation array size'
        return convert_to_np_array(hidden_activations)

    def _get_hidden_error_terms(self, hidden_activations, output_error_terms):
        hidden_error_terms = []
        for num, hidden_activation in enumerate(hidden_activations):
            error_term = hidden_activation * (1 - hidden_activation) * (
                np.dot(output_error_terms, self._hidden_to_output_weights[:, num + 1]))
            hidden_error_terms.append(error_term)
        return np.array(hidden_error_terms, dtype=np.double)

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
