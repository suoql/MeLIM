"""
Implement a couple of methods to measure clustering.
"""

import math
from collections import defaultdict

def data_count(clusters):
    data_num = 0
    for key in clusters.keys():
        data_num = data_num + len(clusters[key])
    return data_num


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke.
    Quote from:
          https://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

class RandIndex:
    """
    (TP + TN) / C(n, 2), where
          TP is the number of pairs of "similar" data that are grouped into the same cluster,
          TN is the number of pairs of "different" data that are grouped into different clusters.
    """

    def _get_cohort_key(self, patient, cohort):
        for key in cohort.keys():
            if patient in cohort[key]:
                return key
        # should throw error here

    def _calculate_tp_for_one_cluster(self, cluster, cohort):
        if len(cluster) < 2:
            return 0

        cohort_key = []
        for i in range(0, len(cluster)):
            cohort_key.append(self._get_cohort_key(cluster[i], cohort))

        tp = 0
        for i in range(0, len(cluster) - 1):
            for j in range(i + 1, len(cluster)):
                if cohort_key[i] == cohort_key[j]:
                    tp = tp + 1
        return tp

    def _calculate_tp(self, clusters, cohort):
        tp = 0
        for cluster_key in clusters.keys():
            tp = tp + self._calculate_tp_for_one_cluster(clusters[cluster_key], cohort)
        return tp

    def _calculate_tn_between_cluster(self, cluster1, cluster2, cohort):
        cohort_key_cluster1 = []
        cohort_key_cluster2 = []
        for i in range(0, len(cluster1)):
            cohort_key_cluster1.append(self._get_cohort_key(cluster1[i], cohort))
        for i in range(0, len(cluster2)):
            cohort_key_cluster2.append(self._get_cohort_key(cluster2[i], cohort))

        tn = 0
        for i in range(0, len(cluster1)):
            for j in range(0, len(cluster2)):
                if cohort_key_cluster1[i] != cohort_key_cluster2[j]:
                    tn = tn + 1
        return tn

    def _calculate_tn(self, clusters, cohort):
        tn = 0
        clusters_keys = list(clusters.keys())
        for i in range(0, len(clusters_keys) - 1):
            for j in range(i + 1, len(clusters_keys)):
                tn = tn + self._calculate_tn_between_cluster(clusters[clusters_keys[i]], clusters[clusters_keys[j]], cohort)
        return tn

    def evaluate(self, clusters, cohort):
        """
        See the class docstring
        clusters: {"clustering ID":[patient ID, patient ID, ..], }
        cohort: {"disease ID": [patient ID, patient ID, ..], ..}
        """
        TP = self._calculate_tp(clusters, cohort)
        TN = self._calculate_tn(clusters, cohort)

        data_num = data_count(clusters)
        choose_two_from_n = choose(data_num, 2)

        ri = 1.0 * (TP + TN) / choose_two_from_n
        return ri


class Purity():
    """
    to do:
    """
    def _intersection(self, array1, array2):
        intersection = [val for val in array1 if val in array2]
        return len(intersection)

    def _max_intersection(self, cluster, cohort):
        max_intersection = 0
        for key in cohort.keys():
            max_intersection = max(max_intersection, self._intersection(cluster, cohort[key]))
        return max_intersection

    def evaluate(self, clusters, cohort):
        """
        clusters: {"clustering ID":[patient ID, patient ID, ..], }
        cohort: {"disease ID": [patient ID, patient ID, ..], ..}
        """
        purity = 0
        for key in clusters.keys():
            purity = purity + self._max_intersection(clusters[key], cohort)

        data_num = data_count(clusters)

        return 1.0 * purity / data_num



def convert_list_2_dict(arr):
    clusters = defaultdict(list)
    for id, label in enumerate(arr):
        clusters[label].append(id)
    return clusters

# refer to:
# https://stats.stackexchange.com/questions/89030/rand-index-calculation
def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)



