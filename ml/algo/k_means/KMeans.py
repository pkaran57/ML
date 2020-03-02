from itertools import starmap

import numpy as np


class KMeans:

    find_euclidean_distance = lambda point_a, point_b: np.linalg.norm(point_a-point_b)

    def __init__(self, training_samples, validation_samples):
        self._training_samples = training_samples
        self._validation_samples = validation_samples

    def find_clusters(self, num_clusters, times_itr):
        centroids = [np.random.rand(self._training_samples[0].features.shape[0]) for _ in range(num_clusters)]
        previous_centroid = [centroid for centroid in centroids]

        for _ in range(times_itr):

            groups = {cluster_num: list() for cluster_num in range(num_clusters)}

            for sample in self._training_samples:
                distances = [KMeans.find_euclidean_distance(sample.features, centroids[cluster_num]) for cluster_num in range(num_clusters)]
                closest_to_centroid = distances.index(min(distances))
                groups[closest_to_centroid].append(sample)

            previous_centroid = [centroid for centroid in centroids]

            for cluster_num, samples in groups.items():
                if samples:
                    np_array = np.array([sample.features for sample in samples])
                    centroids[cluster_num] = np_array.sum(axis=0) / len(samples)

            print('Difference in centroids from previous iteration = ',
                  list(starmap(lambda centroid, previous_centroid: KMeans.find_euclidean_distance(centroid, previous_centroid), zip(centroids, previous_centroid))))
