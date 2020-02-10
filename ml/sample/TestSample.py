class TestSample:
    """
    Class to help create test samples for testing
    """

    def __init__(self, sample_number, true_class_label, inputs):
        self.sample_number = sample_number
        self.true_class_label = int(true_class_label)
        self.inputs = inputs

    def __str__(self):
        return "Training sample #{} : true class label = {}, inputs = {}".format(self.sample_number, self.true_class_label,
                                                                        self.inputs)

    def get_features(self):
        return self.inputs[1:]
