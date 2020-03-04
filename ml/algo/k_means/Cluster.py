import math
from collections import Counter

import numpy as np


class Cluster:

    def __init__(self, num_dimensions):
        self.centroid = np.random.rand(num_dimensions)
        self.samples_in_cluster = list()
        self._centroid_from_previous_iteration = None

    def find_distance_from_centroid(self, point):
        """
        :param point: a point with the same dimensions as the centroid
        :return: Distance between point and current centroid
        """
        return np.linalg.norm(point - self.centroid)

    def update_samples_in_cluster(self, samples):
        if samples and len(samples) >= 1:
            self.samples_in_cluster = samples

            np_array = np.array([sample.features for sample in samples])
            new_centroid = np_array.sum(axis=0) / len(samples)

            self._centroid_from_previous_iteration = self.centroid
            self.centroid = new_centroid
        else:
            self._centroid_from_previous_iteration = self.centroid

    def distance_between_current_and_prev_centroid(self):
        """
        :return: Distance between current centroid and centroid from previous iteration
        """
        if self._centroid_from_previous_iteration is not None:
            return self.find_distance_from_centroid(self._centroid_from_previous_iteration)
        else:
            return None

    def mean_square_error(self):
        if self.samples_in_cluster:
            return sum(map(lambda sample: (self.find_distance_from_centroid(sample.features) ** 2), self.samples_in_cluster)) / len(self.samples_in_cluster)
        else:
            return 0

    def entropy(self):
        entropy_value = 0

        num_samples = len(self.samples_in_cluster)
        samples_count_by_class_labels = Counter([sample.true_class_label for sample in self.samples_in_cluster])

        for true_class_label, count in samples_count_by_class_labels.items():
            x = count / num_samples
            entropy_value += (x * math.log(x, 2))

        return -entropy_value

    def most_common_label(self):
        samples_count_by_class_labels = Counter([sample.true_class_label for sample in self.samples_in_cluster])
        if len(samples_count_by_class_labels) >= 1:
            return samples_count_by_class_labels.most_common(1)[0][0]
        else:
            return None
