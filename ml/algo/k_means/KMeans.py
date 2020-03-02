import numpy as np


class KMeans:

    find_euclidean_distance = lambda point_a, point_b: np.linalg.norm(point_a-point_b)

    def __init__(self, training_samples, validation_samples):
        self._training_samples = training_samples
        self._validation_samples = validation_samples

    def find_clusters(self, num_clusters):
        centroids = [np.random.rand(self._training_samples[0].features.shape[0]) for _ in range(num_clusters)]

        for sample in self._training_samples:
            distances = [KMeans.find_euclidean_distance(sample.features, centroids[cluster_num]) for cluster_num in range(num_clusters)]
            print(distances)
