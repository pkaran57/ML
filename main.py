# Name - Karan Patel, PSU ID - 965051876
# Instructions - Entry point for running the algorithm. Install all the necessary requirements specified in the 'requirements.txt' file by running 'pip install -r requirements.txt'
# Runtime used - Python 3.7

import logging

from ml.algo.neural_net.SingleLaterNeuralNet import SingleLaterNeuralNet
from ml.algo.perceptron.PerceptronLearningAlgo import PerceptronLearningAlgo
from ml.sample.MNISTSample import MNISTSample
from ml.utils.SampleUtils import get_fraction_of_equally_distributed_samples, get_training_samples_by_target_labels

logging.basicConfig(format="'%(asctime)s' %(name)s %(message)s'", level=logging.INFO)
logger = logging.getLogger("MAIN")

# load training and validation samples
training_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_train.csv')
validation_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_test.csv')


def neural_net():
    num_target_labels = 10
    learning_rate = 0.1

    # exp 1
    for num_hidden_units in 20, 50, 100:
        single_layer_neural_net = SingleLaterNeuralNet(num_hidden_units, training_samples, validation_samples, num_target_labels)
        single_layer_neural_net.train(num_of_epochs=50, learning_rate=learning_rate, momentum=0.0)

    # exp 2
    training_samples_by_target_labels = get_training_samples_by_target_labels(training_samples)
    for training_sample_fraction in .5, .25:
        training_samples_subset = get_fraction_of_equally_distributed_samples(num_target_labels,
                                                                              len(training_samples),
                                                                              training_sample_fraction,
                                                                              training_samples_by_target_labels)

        single_layer_neural_net = SingleLaterNeuralNet(100, training_samples_subset, validation_samples, num_target_labels)
        single_layer_neural_net.train(num_of_epochs=50, learning_rate=learning_rate, momentum=0.0)

    # exp 3
    for momentum in 0.25, 0.5, 0.95:
        single_layer_neural_net = SingleLaterNeuralNet(100, training_samples, validation_samples, num_target_labels)
        single_layer_neural_net.train(num_of_epochs=50, learning_rate=learning_rate, momentum=momentum)


def perceptron():
    num_of_epochs = 2
    true_class_labels_in_dataset = set(range(10))
    # For each learning rate, execute the Perceptron learning algorithm and determining accuracy after each epoch and accuracy matrix at the end
    for learning_rate in 0.1, 0.01, .001:
        perceptron_learning_algo = PerceptronLearningAlgo(learning_rate, num_of_epochs, training_samples,
                                                          validation_samples, true_class_labels_in_dataset)
        perceptron_learning_algo.train_and_compute_accuracy()


neural_net()
