import math

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ml.algo.k_means.Cluster import Cluster


class KMeans:

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

        return self.mean_square_error(clusters), clusters

    @staticmethod
    def mean_square_error(clusters):
        return sum(map(Cluster.mean_square_error, clusters)) / len(clusters)

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

        assert len(self._training_samples) == sum(map(len, samples_by_clusters.values())), 'Expected total number of training samples to match the sum of samples in all clusters'
        return samples_by_clusters

    @staticmethod
    def mean_square_separation(clusters):
        mss = 0

        for i in range(len(clusters)):
            for j in range(i):
                if i != j:
                    mss += (clusters[i].find_distance_from_centroid(clusters[j].centroid) ** 2)

        return mss / ((len(clusters) * (len(clusters) - 1)) / 2)

    def mean_entropy(self, clusters):
        num_training_samples = len(self._training_samples)

        mean_entropy_value = 0

        for cluster in clusters:
            mean_entropy_value += ((len(cluster.samples_in_cluster) / num_training_samples) * cluster.entropy())

        return mean_entropy_value

    def compute_accuracy(self, clusters):
        predicted_labels = []
        actual_labels = []
        correct_predictions = 0

        for validation_sample in self._validation_samples:
            distance_cluster_map = {cluster.find_distance_from_centroid(validation_sample.features):cluster for cluster in clusters}
            cluster_for_sample = distance_cluster_map[min(distance_cluster_map.keys())]

            predicted_label = cluster_for_sample.most_common_label()
            actual_label = validation_sample.true_class_label

            if predicted_label == actual_label:
                correct_predictions += 1

            predicted_labels.append(predicted_label)
            actual_labels.append(actual_label)

        validation_samples_confusion_matrix = confusion_matrix(y_true=actual_labels, y_pred=predicted_labels)

        confusion_matrix_display = ConfusionMatrixDisplay(validation_samples_confusion_matrix, range(10))
        confusion_matrix_display.plot(values_format='d')

        plt.title('Confusion matrix')

        plt.savefig('out/confusion-matrix.png', format='png', dpi=1200)
        plt.show()

        accuracy = (correct_predictions / len(self._validation_samples)) * 100
        print('Accuracy = {}%'.format(accuracy))

    @staticmethod
    def print_centroid(clusters):
        for num, cluster in enumerate(clusters):
            plt.imshow(cluster.centroid.reshape(8,8), cmap='Greys')

            common_label = cluster.most_common_label()
            plt.title('No samples in cluster' if common_label is None else 'Samples with the most common class label in this cluster = {}'.format(common_label))

            plt.savefig('out/{}-centroid-visual.png'.format(num), format='png', dpi=1200)
            plt.show()
