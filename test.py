import math

import numpy as np

from algo.neural_net.SingleLaterNeuralNet import SingleLaterNeuralNet
from sample.TestSample import TestSample


def test_neural_net():

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

    assert math.isclose(input_to_hidden_weights[0][0], -.4002, abs_tol=0.0001)
    assert math.isclose(input_to_hidden_weights[0][1], .1998, abs_tol=0.001)
    assert math.isclose(input_to_hidden_weights[0][2], .1, abs_tol=0.0001)

    assert math.isclose(input_to_hidden_weights[1][0], -.20006, abs_tol=0.0001)
    assert math.isclose(input_to_hidden_weights[1][1], .39994, abs_tol=0.001)
    assert math.isclose(input_to_hidden_weights[1][2], -.1, abs_tol=0.0001)

