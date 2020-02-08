"""
Name - Karan Patel, PSU ID - 965051876

Class that represents an instance of a Perceptron Learning algorithm
"""

import logging
import matplotlib.pyplot as plt

from collections import Counter
from domain.Perceptron import Perceptron
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class PerceptronLearningAlgo:

    def __init__(self, learning_rate, num_of_epochs, training_samples, validation_samples,
                 true_class_labels_in_dataset):
        self.__logger = logging.getLogger('learning-rate={}'.format(learning_rate))

        self.__learning_rate = learning_rate
        self.__num_of_epochs = num_of_epochs
        self.__perceptrons = [Perceptron(training_samples, perceptron_class_label) for perceptron_class_label in
                              true_class_labels_in_dataset]

        self.__training_samples = training_samples
        self.__training_accuracies_per_epoch = []

        self.__validation_samples = validation_samples
        self.__validation_accuracies_per_epoch = []

        self.__validation_sample_count_by_labels = Counter([sample.true_class_label for sample in validation_samples])

    def train_and_compute_accuracy(self):
        """
        Train all perceptrons (each belonging to a unique target label) for a given number of epochs.
        Determining accuracy for training and validation samples after each epoch and accuracy matrix at the end. Generate diagrams for both.
        :return:
        """
        self.__compute_accuracy(0)

        for epoch_num in range(1, self.__num_of_epochs + 1):
            list(map(lambda perceptron: perceptron.train_for_an_epoch(self.__learning_rate), self.__perceptrons))
            self.__compute_accuracy(epoch_num)

        self.__plot_accuracies()
        self.__compute_confusion_matrix()

    def __compute_accuracy(self, epoch_num):
        """
        Compute accuracy for training and validation samples after each epoch
        :param epoch_num:
        :return:
        """
        for samples in self.__validation_samples, self.__training_samples:
            correct_predictions = 0
            total_num_predictions = len(samples)

            for sample in samples:
                perceptron_target_label_with_max_net_input = self.__get_prediction_for_sample(sample)

                if perceptron_target_label_with_max_net_input == sample.true_class_label:
                    correct_predictions += 1

            accuracy = (correct_predictions / total_num_predictions) * 100

            if samples is self.__validation_samples:
                self.__validation_accuracies_per_epoch.append(accuracy)
            else:
                self.__training_accuracies_per_epoch.append(accuracy)

            self.__logger.info("For epoch #{}, accuracy for {} dataset = {}".format(epoch_num,
                                                                                    'training' if samples is self.__training_samples else 'validation',
                                                                                    accuracy))

    def __get_prediction_for_sample(self, sample):
        """
        Get prediction for sample
        :param sample: Sample's class to predict
        :return: Prediction of the algorithm for the sample
        """
        net_inputs = list(map(lambda perceptron: perceptron.get_net_input(sample), self.__perceptrons))
        perceptron_target_label_with_max_net_input = net_inputs.index(max(net_inputs))
        return perceptron_target_label_with_max_net_input

    def __plot_accuracies(self):
        plt.title("Accuracy over epochs for learning rate = {}".format(self.__learning_rate))

        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy (%)')

        plt.plot(self.__training_accuracies_per_epoch, label='Training Accuracy')
        plt.plot(self.__validation_accuracies_per_epoch, label='Validation Accuracy')

        plt.legend()
        plt.show()

    def __compute_confusion_matrix(self):
        predicted_labels = []
        actual_labels = []

        for validation_sample in self.__validation_samples:
            predicted_target_label = self.__get_prediction_for_sample(validation_sample)
            predicted_labels.append(predicted_target_label)
            actual_labels.append(validation_sample.true_class_label)

        training_samples_confusion_matrix = confusion_matrix(y_true=actual_labels, y_pred=predicted_labels)

        for true_class_label, row in enumerate(training_samples_confusion_matrix):
            actual_count = sum(row)
            expected_count = self.__validation_sample_count_by_labels.get(true_class_label)

            assert actual_count == expected_count, \
                'Total sample count for true class label {} does not equal to the count in confusion matrix. Expected = {}, Actual = {}'.format(
                    true_class_label,
                    expected_count,
                    actual_count)

        confusion_matrix_display = ConfusionMatrixDisplay(training_samples_confusion_matrix, range(10))
        confusion_matrix_display.plot(values_format='d')
        plt.title("Confusion matrix for learning rate = {}".format(self.__learning_rate))
        plt.show()

        self.__logger.info("Confusion matrix = \n\n{}\n\n".format(training_samples_confusion_matrix))
