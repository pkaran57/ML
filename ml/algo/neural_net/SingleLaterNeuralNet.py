import logging
import math
import random

import numpy as np

from ml.utils import ReportUtils, PlotUtils
from ml.utils.ArrayUtils import convert_to_np_array


class SingleLaterNeuralNet:
    """
    Neural net with a single hidden layer
    """

    def __init__(self, num_hidden_units, training_samples, validation_samples, num_target_labels, **kwargs):
        """

        :param num_hidden_units: number of hidden units to create in the hidden later
        :param training_samples:
        :param validation_samples:
        :param num_target_labels: total number of truth labels
        :param kwargs: Additional options to override weight creation process. Use 'input_to_hidden_weights' and 'hidden_to_output_weights' to provide desired weights.
                       If not provided, random small weights will be used to initialize both weight matrices.
        """
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

        self._training_accuracies_over_epoch = []
        self._validation_accuracies_over_epoch = []

        self._logger = logging.getLogger('NeuralNet')

    def train(self, num_of_epochs, learning_rate=0.1, momentum=0):
        """
        Train neural net with training samples. Compute training as well as validation accuracies before training as well as after each epoch during training.
        Accuracy over epochs and a confusion matrix against the validation dataset will be generated during the training.
        :param num_of_epochs:
        :param learning_rate:
        :param momentum:
        :return:
        """
        title_attributes = {'η': learning_rate, 'α': momentum, '# of hidden units': self._num_hidden_units, '# of training samples': len(self._training_samples)}
        self.compute_accuracy(0)

        for epoch_num in range(1, num_of_epochs + 1):
            self._logger.info('Starting training for epoch #{}'.format(epoch_num))
            self._train_for_an_epoch(learning_rate, momentum)

            self.compute_accuracy(epoch_num)

        PlotUtils.plot_accuracy(self._training_accuracies_over_epoch, self._validation_accuracies_over_epoch, title_attributes)
        # line below plots accuracy without
        PlotUtils.plot_accuracy(self._training_accuracies_over_epoch[1:], self._validation_accuracies_over_epoch[1:], title_attributes)
        PlotUtils.plot_confusion_matrix(self._validation_samples, self.get_prediction, title_attributes)

    def compute_accuracy(self, epoch_num):
        """
        Compute accuracy against training and  validation samples
        :param epoch_num: Only used for logging
        :return:
        """
        for samples in self._training_samples, self._validation_samples:
            is_validation_sample = True if samples is self._validation_samples else False
            accuracy = ReportUtils.compute_accuracy(samples, self.is_prediction_correct)

            self._validation_accuracies_over_epoch.append(accuracy) if is_validation_sample else self._training_accuracies_over_epoch.append(accuracy)
            self._logger.info("For epoch #{}, accuracy for {} dataset = {}".format(epoch_num, 'validation samples' if is_validation_sample else 'training samples', accuracy))

    def is_prediction_correct(self, sample):
        return self.get_prediction(sample) == sample.true_class_label

    def get_prediction(self, sample):
        """
        :param sample: sample's class label to predict
        :return: predicted class label for sample
        """
        hidden_activations = self._get_hidden_activations(sample)
        output_activations = self._get_output_activations(hidden_activations)
        return output_activations.argmax()

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
