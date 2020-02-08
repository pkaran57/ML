# Name - Karan Patel, PSU ID - 965051876
# Instructions - Entry point for running the Perceptron learning algorithm. Install all the necessary requirements specified in the 'requirements.txt' file by running 'pip install -r requirements.txt'
# Runtime used - Python 3.7

import logging

from sample.MNISTSample import MNISTSample
from domain.PerceptronLearningAlgo import PerceptronLearningAlgo

logging.basicConfig(format="'%(asctime)s' %(name)s %(message)s'", level=logging.INFO)
logger = logging.getLogger("MAIN")

# load training and validation samples
training_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_train.csv')
validation_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_test.csv')

num_of_epochs = 50
true_class_labels_in_dataset = set(range(10))

# For each learning rate, execute the Perceptron learning algorithm and determining accuracy after each epoch and accuracy matrix at the end
for learning_rate in 0.1, 0.01, .001:
    perceptron_learning_algo = PerceptronLearningAlgo(learning_rate, num_of_epochs, training_samples, validation_samples, true_class_labels_in_dataset)
    perceptron_learning_algo.train_and_compute_accuracy()
