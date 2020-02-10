import math

import numpy as np

from ml.algo.neural_net.SingleLaterNeuralNet import SingleLaterNeuralNet
from ml.sample.TestSample import TestSample


def test_train(monkeypatch):
    training_samples = [TestSample(1, 0, np.array([1.0, 1.0, 0.0], dtype=np.double))]
    input_to_hidden_weights = np.array([[-0.4, 0.2, 0.1], [-0.2, 0.4, -0.1]], dtype=np.double)
    hidden_to_output_weights = np.array([[.1, -.2, .1], [.4, -.1, .1]], dtype=np.double)

    monkeypatch.setattr('ml.algo.neural_net.SingleLaterNeuralNet.SingleLaterNeuralNet.compute_accuracy', lambda self, epoch_num: None)
    monkeypatch.setattr('ml.utils.PlotUtils.plot_confusion_matrix', lambda x1, x2, x3: None)
    monkeypatch.setattr('ml.utils.PlotUtils.plot_accuracy', lambda x1, x2, x3: None)

    single_layer_neural_net = SingleLaterNeuralNet(2, training_samples, None, 2,
                                                   input_to_hidden_weights=input_to_hidden_weights,
                                                   hidden_to_output_weights=hidden_to_output_weights)

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


def test__calculate_hidden_to_output_delta_zero_delta():
    learning_rate = .1
    momentum = .9
    hidden_activations = [.45, .55]
    output_error_terms = np.array([.095, -.12], dtype=np.double)

    hidden_to_output_weights_delta_from_previous_itr = np.zeros(shape=(2, 3))

    hidden_to_output_delta = SingleLaterNeuralNet._calculate_hidden_to_output_delta(hidden_activations, learning_rate,
                                                                                    momentum, output_error_terms,
                                                                                    hidden_to_output_weights_delta_from_previous_itr)

    assert hidden_to_output_delta.shape == (2, 3)

    abs_tol = 0.001

    for actual, expected in zip(hidden_to_output_delta[0], [0.01, .004, .005]):
        assert math.isclose(actual, expected, abs_tol=abs_tol)

    for actual, expected in zip(hidden_to_output_delta[1], [-.012, -.0054, -.0066]):
        assert math.isclose(actual, expected, abs_tol=abs_tol)


def test__calculate_hidden_to_output_delta_non_zero_delta():
    learning_rate = .1
    momentum = 1
    hidden_activations = [.45, .55]
    output_error_terms = np.array([.095, -.12], dtype=np.double)

    hidden_to_output_weights_delta_from_previous_itr = np.array([[.007, .2, -.1], [0, 0, 1]], dtype=np.float)

    hidden_to_output_delta = SingleLaterNeuralNet._calculate_hidden_to_output_delta(hidden_activations, learning_rate,
                                                                                    momentum, output_error_terms,
                                                                                    hidden_to_output_weights_delta_from_previous_itr)

    assert hidden_to_output_delta.shape == (2, 3)

    abs_tol = 0.001

    for actual, expected in zip(hidden_to_output_delta[0], [0.017, .204, -0.095]):
        assert math.isclose(actual, expected, abs_tol=abs_tol)

    for actual, expected in zip(hidden_to_output_delta[1], [-.012, -.0054, 0.9934]):
        assert math.isclose(actual, expected, abs_tol=abs_tol)
