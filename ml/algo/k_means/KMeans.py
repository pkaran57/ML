import math

import numpy as np

from ml.algo.k_means.Cluster import Cluster


class KMeans:

    find_euclidean_distance = lambda point_a, point_b: np.linalg.norm(point_a - point_b)

    def __init__(self, training_samples, validation_samples):
        self._training_samples = training_samples
        self._validation_samples = validation_samples

    def find_clusters(self, num_clusters):
        # number of dimensions for the centroid
        centroid_num_dimensions = self._training_samples[0].features.shape[0]
        clusters = [Cluster(centroid_num_dimensions) for _ in range(num_clusters)]
        distance_between_current_and_prev_centroid_for_clusters = [9999999999 for _ in range(len(clusters))]

        iter_num = 0

        while not all(map(lambda distance: math.isclose(distance, 0, abs_tol=0.001), distance_between_current_and_prev_centroid_for_clusters)):

            samples_by_clusters = self.group_samples_by_clusters(clusters, self._training_samples)

            for cluster, samples in samples_by_clusters.items():
                cluster.update_samples_in_cluster(samples)

            distance_between_current_and_prev_centroid_for_clusters = list(map(Cluster.distance_between_current_and_prev_centroid, clusters))

            print('Itr num = {}, difference in centroids from previous iteration = {}'.format(iter_num,
                                                                                              distance_between_current_and_prev_centroid_for_clusters))

            iter_num += 1

        print('Average mean square error = ', sum(map(Cluster.mean_square_error, clusters)) / len(clusters))
        print('Mean square seperation = ', self.mean_square_separation(clusters))


    def group_samples_by_clusters(self, clusters, samples):
        """
        Group samples by clusters
        :param clusters: distinct clusters to group samples by
        :return: a map of cluster to list of samples belonging to that cluster
        """
        samples_by_clusters = {cluster: list() for cluster in clusters}

        for sample in samples:
            distance_cluster_map = {cluster.find_distance_from_centroid(sample.features):cluster for cluster in clusters}
            cluster_for_sample = distance_cluster_map[min(distance_cluster_map.keys())]
            samples_by_clusters[cluster_for_sample].append(sample)
        return samples_by_clusters

    def mean_square_separation(self, clusters):
        mss = 0

        for main_cluster in clusters:
            for cluster in clusters:
                if main_cluster != cluster:
                    mss += (main_cluster.find_distance_from_centroid(cluster.centroid) ** 2)

        return mss / ((len(clusters) * (len(clusters) - 1)) / 2)

    def mean_entropy(self):
        ...

