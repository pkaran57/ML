import math
from unittest import TestCase

import numpy as np

from ml.algo.neural_net.SingleLaterNeuralNet import SingleLaterNeuralNet
from ml.sample.TestSample import TestSample


class TestSingleLaterNeuralNet(TestCase):

    def test_train(self):
        training_samples = [TestSample(1, 0, np.array([1.0, 1.0, 0.0], dtype=np.double))]
        input_to_hidden_weights = np.array([[-0.4, 0.2, 0.1], [-0.2, 0.4, -0.1]], dtype=np.double)
        hidden_to_output_weights = np.array([[.1, -.2, .1], [.4, -.1, .1]], dtype=np.double)

        single_layer_neural_net = SingleLaterNeuralNet(2, training_samples, None, 2, input_to_hidden_weights=input_to_hidden_weights, hidden_to_output_weights=hidden_to_output_weights)

        # method under test
        single_layer_neural_net.train(num_of_epochs=1, learning_rate=0.1, momentum=0.9)

        input_to_hidden_weights = single_layer_neural_net._input_to_hidden_weights
        hidden_to_output_weights = single_layer_neural_net._hidden_to_output_weights

        assert input_to_hidden_weights.shape == (2, 3)
        assert hidden_to_output_weights.shape == (2, 3)

        abs_tol = 0.0001

        for actual, expected in zip(input_to_hidden_weights[0], [-.4002, .1998, .1]):
            assert math.isclose(actual, expected, abs_tol=abs_tol)

        for actual, expected in zip(input_to_hidden_weights[1], [-.20006, .39994, -.1]):
            assert math.isclose(actual, expected, abs_tol=abs_tol)
