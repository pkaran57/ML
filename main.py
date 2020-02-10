# Name - Karan Patel, PSU ID - 965051876
# Instructions - Entry point for running the Perceptron learning algorithm. Install all the necessary requirements specified in the 'requirements.txt' file by running 'pip install -r requirements.txt'
# Runtime used - Python 3.7

import logging

from ml.algo.neural_net.SingleLaterNeuralNet import SingleLaterNeuralNet
from ml.algo.perceptron.PerceptronLearningAlgo import PerceptronLearningAlgo
from ml.sample.MNISTSample import MNISTSample

logging.basicConfig(format="'%(asctime)s' %(name)s %(message)s'", level=logging.INFO)
logger = logging.getLogger("MAIN")

# load training and validation samples
training_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_train.csv')
validation_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_test.csv')


def neural_net():
    for num_hidden_units in 20, 50, 100:
        single_layer_neural_net = SingleLaterNeuralNet(num_hidden_units, training_samples, validation_samples, 10)
        single_layer_neural_net.train(num_of_epochs=50, learning_rate=0.1, momentum=0.0)


def perceptron():
    num_of_epochs = 2
    true_class_labels_in_dataset = set(range(10))
    # For each learning rate, execute the Perceptron learning algorithm and determining accuracy after each epoch and accuracy matrix at the end
    for learning_rate in 0.1, 0.01, .001:
        perceptron_learning_algo = PerceptronLearningAlgo(learning_rate, num_of_epochs, training_samples,
                                                          validation_samples, true_class_labels_in_dataset)
        perceptron_learning_algo.train_and_compute_accuracy()


neural_net()
