# Name - Karan Patel, PSU ID - 965051876
# Instructions - Entry point for running the algorithm. Install all the necessary requirements specified in the 'requirements.txt' file by running 'pip install -r requirements.txt'
# Runtime used - Python 3.7

import logging

from ml.algo.k_means.KMeans import KMeans
from ml.algo.naive_bayes.naive_bayes import naive_bayes
from ml.algo.neural_net.SingleLaterNeuralNet import SingleLaterNeuralNet
from ml.algo.perceptron.PerceptronLearningAlgo import PerceptronLearningAlgo
from ml.sample.MNISTSample import MNISTSample
from ml.sample.OptDigitSample import OptDigitSample
from ml.utils.SampleUtils import get_fraction_of_equally_distributed_samples, get_training_samples_by_target_labels

logging.basicConfig(format="'%(asctime)s' %(name)s %(message)s'", level=logging.INFO)
logger = logging.getLogger("MAIN")


def neural_net_main():
    training_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_train.csv')
    validation_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_test.csv')

    num_target_labels = 10
    learning_rate = 0.1

    # exp 1
    for num_hidden_units in 20, 50, 100:
        single_layer_neural_net = SingleLaterNeuralNet(num_hidden_units, training_samples, validation_samples,
                                                       num_target_labels)
        single_layer_neural_net.train(num_of_epochs=50, learning_rate=learning_rate, momentum=0.0)

    # exp 2
    training_samples_by_target_labels = get_training_samples_by_target_labels(training_samples)
    for training_sample_fraction in .5, .25:
        training_samples_subset = get_fraction_of_equally_distributed_samples(num_target_labels,
                                                                              len(training_samples),
                                                                              training_sample_fraction,
                                                                              training_samples_by_target_labels)

        single_layer_neural_net = SingleLaterNeuralNet(100, training_samples_subset, validation_samples,
                                                       num_target_labels)
        single_layer_neural_net.train(num_of_epochs=50, learning_rate=learning_rate, momentum=0.0)

    # exp 3
    for momentum in 0.25, 0.5, 0.95:
        single_layer_neural_net = SingleLaterNeuralNet(100, training_samples, validation_samples, num_target_labels)
        single_layer_neural_net.train(num_of_epochs=50, learning_rate=learning_rate, momentum=momentum)


def perceptron_main():
    training_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_train.csv')
    validation_samples = MNISTSample.load_and_shuffle_samples_from_dataset('data/mnist_test.csv')

    num_of_epochs = 2
    true_class_labels_in_dataset = set(range(10))
    # For each learning rate, execute the Perceptron learning algorithm and determining accuracy after each epoch and accuracy matrix at the end
    for learning_rate in 0.1, 0.01, .001:
        perceptron_learning_algo = PerceptronLearningAlgo(learning_rate, num_of_epochs, training_samples,
                                                          validation_samples, true_class_labels_in_dataset)
        perceptron_learning_algo.train_and_compute_accuracy()


def naive_bayes_main():
    training_file = 'C:\K.E.R Projects\ml\data\yeast_training.txt'
    test_file = 'C:\K.E.R Projects\ml\data\yeast_test.txt'

    naive_bayes(training_file, test_file)


def k_means_main(num_clusters=10):
    training_samples = OptDigitSample.load_and_shuffle_samples_from_dataset('data/optdigits/optdigits.train')
    test_samples = OptDigitSample.load_and_shuffle_samples_from_dataset('data/optdigits/optdigits.test')

    k_means_algo = KMeans(training_samples, test_samples)
    runs = dict()

    for run_num in range(5):
        mse, clusters = k_means_algo.find_clusters(num_clusters)
        print("Mean square for run #{} = {}".format(run_num + 1, mse))
        runs[mse] = clusters

    smallest_mse = min(runs.keys())
    print("Using cluster with the following smallest mse - {}".format(smallest_mse))
    cluster_set = runs[smallest_mse]

    print('Average mean square error = ', KMeans.mean_square_error(cluster_set))
    print('Mean square separation = ', KMeans.mean_square_separation(cluster_set))
    print('Mean entropy = ', k_means_algo.mean_entropy(cluster_set))

    k_means_algo.compute_accuracy(cluster_set)
    k_means_algo.print_centroid(cluster_set)


k_means_main()
