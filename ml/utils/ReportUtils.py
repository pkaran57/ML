import logging

_logger = logging.getLogger('ReportUtils')


def compute_accuracy(samples, prediction_function):
    total_num_predictions = len(samples)
    correct_predictions = list(map(prediction_function, samples)).count(True)
    accuracy = (correct_predictions / total_num_predictions) * 100
    return accuracy
