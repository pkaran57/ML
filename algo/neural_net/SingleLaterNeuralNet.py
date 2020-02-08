import math
import random

import numpy as np


class SingleLaterNeuralNet:

    def __init__(self, num_hidden_units, training_samples, validation_samples, num_target_labels):
        self.__num_hidden_units = num_hidden_units
        self.__training_samples = training_samples
        self.__validation_samples = validation_samples
        self.__num_target_labels = num_target_labels

        self.__input_to_hidden_weights = SingleLaterNeuralNet.initialize_weight_matrix(
            shape=(num_hidden_units, training_samples[0].inputs.size))  # 50 * 785 . 785 * 1 = 50 * 1
        self.__hidden_to_output_weights = SingleLaterNeuralNet.initialize_weight_matrix(
            shape=(num_target_labels, num_hidden_units + 1))  # 10 * 51 *

        self.__input_to_hidden_weights_delta_from_previous_itr = np.zeros(shape=self.__input_to_hidden_weights.shape,
                                                                          dtype=np.double)
        self.__hidden_to_output_weights_delta_from_previous_itr = np.zeros(shape=self.__hidden_to_output_weights.shape,
                                                                           dtype=np.double)

    def train(self, learning_rate=0.1, momentum=0.1):
        for sample in self.__training_samples:
            hidden_activations = list(map(SingleLaterNeuralNet.sigmoid_activation_function,
                                          np.dot(self.__input_to_hidden_weights, sample.inputs)))
            assert len(hidden_activations) == self.__num_hidden_units, 'Unexpected hidden activation array size'

            output_activations = list(map(SingleLaterNeuralNet.sigmoid_activation_function,
                                          np.dot(self.__hidden_to_output_weights, [1.0] + hidden_activations)))
            assert len(output_activations) == self.__num_target_labels, 'Unexpected output activation array size'

            output_error_terms = [
                activation * (1 - activation) * (0.9 if act_num == sample.true_class_label else 0.1 - activation) for
                act_num, activation in enumerate(output_activations)]
            hidden_error_terms = self.get_hidden_error_terms(hidden_activations, output_error_terms)

            hidden_to_output_delta = np.zeros(shape=self.__hidden_to_output_weights.shape, dtype=np.double)
            # Δwk,j=ηδkhj+αΔwʹk,jwhereΔwʹk,jis change to this weight from previous iteration

            self.__hidden_to_output_weights = self.__hidden_to_output_weights + hidden_to_output_delta

            input_to_hidden_delta = np.zeros(shape=self.__input_to_hidden_weights.shape, dtype=np.double)
            self.__input_to_hidden_weights = self.__input_to_hidden_weights + input_to_hidden_delta

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
