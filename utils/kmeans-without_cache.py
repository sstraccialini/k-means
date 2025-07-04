from typing import Union, Literal
import time
from cProfile import Profile
from pstats import SortKey, Stats

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
# from torchvision import datasets
from sklearn import datasets as skdatasets


class KMeans:
    """
    Perform KMeans clustering on a dataset.
    """

    def __init__(self,
                 algorithm : Literal['lloyd', 'extended-hartigan', 'safe-hartigan', 'hartigan', 'binary-hartigan'] = 'lloyd',
                 init : Literal['random', 'random-data', 'k-means++', 'maximin', 'greedy-k-means++'] = 'random',
                 seed : Union[int, None] = None):
        """
        Initialize the KMeans object.

        Parameters
        ----------
        algorithm : {'lloyd', 'extended-hartigan', 'safe-hartigan', 'hartigan', 'binary-hartigan'}
            Algorithm to use. Either 'lloyd' or 'extended-hartigan' or 'safe-hartigan' or 'hartigan' or 'binary-hartigan'

        init : {'random', 'random-data', 'k-means++', 'maximin', 'greedy-k-means++'}
            Initialization method. Either 'random' or 'random-data' or 'k-means++' or 'maximin' or 'greedy-k-means++'

        seed : int
            Seed for random generator
        """

        assert algorithm in ['lloyd', 'extended-hartigan', 'safe-hartigan', 'hartigan', 'binary-hartigan'], "algorithm must be either 'lloyd', 'extended-hartigan', 'safe-hartigan', 'hartigan' or 'binary-hartigan'"
        assert init in ['random', 'random-data', 'k-means++', 'maximin', 'greedy-k-means++'], "init must be either 'random', 'random-data', 'k-means++', 'maximin', or 'greedy-k-means++'"
        assert seed is None or isinstance(seed, int), "seed must be an int or None"

        self.algorithm = algorithm
        self.init = init
        self.seed = seed

        self.data = None
        self.k = None
        self.n_samples = None
        self.centroids = None
        self.y_pred = None

        self.safe_iterations = 0
        self.iterations = 0
        self.norm_calculations = 0

        self._distance_cache = None


    def fit(self, data : np.ndarray, k : int, debug : int = 0):
        """
        Fit the model to the data.

        Parameters
        ----------
        data : np.ndarray
            nxd DataFrame of n samples with d features
        k : int
            Number of clusters
        debug : int
            Debug level (0: no debug, 1: some debug, 2: all debug)

        Returns
        -------
        np.ndarray
            Array of shape (k, d) with cluster centroids
        np.ndarray
            Array of length n with cluster assignments for each sample
        """

        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert len(data.shape) == 2, "data must be a 2D array"
        assert isinstance(k, int), "k must be an int"
        assert 0 < k <= len(data), "k must be at least 0 and at most the number of samples"
        assert isinstance(debug, int) or debug, "debug must be an int"

        self.data = data
        self.k = k
        self.n_samples = data.shape[0]

        np.random.seed(self.seed)

        # initialize centroids
        self._init_centroids(debug)
        debug and print('initial centroids:\n', self.centroids)
        debug and print('initial y_pred:', self.y_pred)

        if self.algorithm == 'lloyd':
            self._lloyd(debug)
        elif self.algorithm == 'extended-hartigan':
            self._extended_hartigan(always_safe=False, debug=debug)
        elif self.algorithm == 'safe-hartigan':
            self._extended_hartigan(always_safe=True, binary_hartigan=False, debug=debug)
        elif self.algorithm == 'binary-hartigan':
            self._extended_hartigan(always_safe=False, binary_hartigan=True, debug=debug)
        elif self.algorithm == 'hartigan':
            self._hartigan(debug)
        
        debug and print('final centroids:\n', self.centroids)
        debug and print('final y_pred:', self.y_pred)


    def _init_centroids(self, debug=0):
        """
        Initialize the centroids.
        """

        v = 1

        if self.init == 'random':

            if v == 1:
                # choose k random data points as initial centroids
                idx = np.random.choice(self.n_samples, self.k, replace=False)
                self.centroids = self.data[idx]
                self.y_pred = self._assign_clusters(debug=debug>1)

            elif v == 2:
                # choose k random data points as initial centroids
                idx = np.random.choice(self.n_samples, self.k, replace=False)
                self.centroids = self.data[idx]

                self._distance_cache = cdist(self.data, self.centroids, metric='sqeuclidean')
                self.y_pred = self._assign_clusters(debug=debug>1)

        elif self.init == 'random-data':
            return NotImplemented
            
            # V1
            # assign each data point to a random cluster
            clusters = np.random.choice(self.k, self.n_samples)

            # check that at least one point is assigned to each cluster
            while len(set(clusters)) < self.k:
                clusters = np.random.choice(self.k, self.n_samples)
            self.y_pred = clusters
            self.centroids = self._move_centroids(None, debug > 1)

            
            # V2
            n_samples, n_features = self.data.shape
            # Assign each data point to a random cluster
            self.y_pred = np.random.choice(self.k, n_samples)
            # Ensure at least one point is assigned to each cluster
            unique_clusters = np.unique(self.y_pred)
            if len(unique_clusters) < self.k:
                for i in range(self.k):
                    if i not in unique_clusters:
                        self.y_pred[np.random.randint(0, n_samples)] = i
            
            self.centroids = np.zeros((self.k, n_features))
            # Calculate centroids efficiently
            for i in range(self.k):
                mask = self.y_pred == i
                if np.any(mask):
                    self.centroids[i] = np.mean(self.data[mask], axis=0)
                else:
                    # Handle empty cluster
                    self.centroids[i] = self.data[np.random.randint(0, n_samples)]

        elif self.init == 'k-means++':
            
            if v == 1:
                # choose first centroid randomly
                centroids = np.zeros((self.k, self.data.shape[1]))
                centroids[0] = self.data[np.random.choice(self.n_samples, 1, replace=False)[0]]
                debug and print('centroids:\n', centroids)

                # iterate over remaining k-1 centroids
                for i in range(1, self.k):
                    debug and print('iteration', i)

                    # calculate squared distance of each point to closest centroid
                    dist = np.min(np.linalg.norm(self.data[:, np.newaxis] - centroids[:i], axis=2) ** 2, axis=1)

                    # probabilities are given by the normalized distance squared
                    probs = dist / dist.sum()
                    debug and print('probs:', probs)

                    # choose next centroid randomly based on cumulated probabilities
                    j = np.random.choice(self.n_samples, p=probs)

                    centroids[i] = self.data[j]
                    debug and print('centroids:\n', centroids)

                self.centroids = centroids
                self.y_pred = self._assign_clusters(debug=debug>1)

            elif v == 2:
                # choose first centroid randomly
                self.centroids = np.zeros((self.k, self.data.shape[1]))
                first_idx = np.random.choice(self.n_samples)
                self.centroids[0] = self.data[first_idx].copy()

                debug and print('centroids:\n', self.centroids)

                # initialize min_distances with infinity
                min_distances = np.full(self.n_samples, np.inf)

                # iterate over remaining k-1 centroids
                for i in range(1, self.k):
                    debug and print('iteration', i)

                    # calculate squared distance of each point to closest centroid (only calculate for new centroids)
                    new_distances = np.sum((self.data - self.centroids[i-1])**2, axis=1)
                    np.minimum(min_distances, new_distances, out=min_distances)

                    # probabilities are given by the normalized distance squared
                    probs = min_distances / np.sum(min_distances)
                    debug and print('probs:', probs)

                    # choose next centroid randomly based on cumulated probabilities
                    j = np.random.choice(self.n_samples, p=probs)

                    self.centroids[i] = self.data[j].copy()
                    debug and print('centroids:\n', self.centroids)

                self._distance_cache = cdist(self.data, self.centroids, metric='sqeuclidean')
                self.y_pred = np.argmin(self._distance_cache, axis=1)

        elif self.init == 'maximin':

            if v == 1:
                # choose first centroid randomly
                centroids = np.zeros((self.k, self.data.shape[1]))
                centroids[0] = self.data[np.random.choice(self.n_samples, 1, replace=False)[0]]
                debug and print('centroids:\n', centroids)

                # iterate over remaining k-1 centroids
                for i in range(1, self.k):
                    debug and print('iteration', i)

                    # calculate squared distance of each point to closest centroid
                    dist = np.min(np.linalg.norm(self.data[:, np.newaxis] - centroids[:i], axis=2) ** 2, axis=1)

                    # choose next centroid as the point with the maximum distance to the closest centroid
                    centroids[i] = self.data[np.argmax(dist)]
                    debug and print('centroids:\n', centroids)

                self.centroids = centroids
                self.y_pred = self._assign_clusters(debug=debug>1)

            elif v == 2:
                # choose first centroid randomly
                self.centroids = np.zeros((self.k, self.data.shape[1]))
                first_idx = np.random.choice(self.n_samples)
                self.centroids[0] = self.data[first_idx].copy()

                # initialize min_distances with infinity
                min_distances = np.full(self.n_samples, np.inf)

                # iterate over remaining k-1 centroids
                for i in range(1, self.k):
                    debug and print('iteration', i)

                    # calculate squared distance of each point to closest centroid (only calculate for new centroids)
                    new_distances = np.sum((self.data - self.centroids[i-1])**2, axis=1)
                    np.minimum(min_distances, new_distances, out=min_distances) 

                    # choose next centroid as the point with the maximum min_distance to the closest centroid
                    self.centroids[i] = self.data[np.argmax(min_distances)].copy()
                    debug and print('centroids:\n', self.centroids)

                self._distance_cache = cdist(self.data, self.centroids, metric='sqeuclidean')
                self.y_pred = np.argmin(self._distance_cache, axis=1)

        elif self.init == 'greedy-k-means++':

            if v == 1:
                # TODO: this might be adapted
                trials = 2 + int(np.log(self.k)) 

                # choose first centroid randomly
                centroids = np.zeros((self.k, self.data.shape[1]))
                f = np.random.choice(self.n_samples)
                centroids[0] = self.data[f]
                debug and print('centroids:\n', centroids)

                # iterate over remaining k-1 centroids
                for i in range(1, self.k):
                    debug and print('iteration', i)

                    # calculate squared distance of each point to closest centroid
                    dist = np.min(np.linalg.norm(self.data[:, np.newaxis] - centroids[:i], axis=2) ** 2, axis=1)

                    # probabilities are given by the normalized distance squared
                    probs = dist / dist.sum()
                    debug and print('probs:', probs)

                    # choose next centroid randomly based on cumulated probabilities
                    candidate_j = np.random.choice(self.n_samples, size=trials, p=probs)
                    debug and print('candidate_j:', candidate_j)

                    # calculate the cost of each trial
                    costs = np.zeros(trials)
                    for t in range(trials):
                        centroids[i] = self.data[candidate_j[t]]
                        costs[t] = self._tot_cluster_cost(centroids, self._assign_clusters(specific_centroids=centroids), debug=debug>1)
                    debug and print('costs:', costs)

                    # choose the trial with the lowest cost
                    j = candidate_j[np.argmin(costs)]

                    centroids[i] = self.data[j]
                    debug and print('centroids:\n', centroids)

                self.centroids = centroids
                self.y_pred = self._assign_clusters(debug=debug>1)

            elif v == 2:
                # V2
                # TODO: this might be adapted
                trials = 2 + int(np.log(self.k)) 

                # choose first centroid randomly
                self.centroids = np.zeros((self.k, self.data.shape[1]))
                first_idx = np.random.choice(self.n_samples)
                self.centroids[0] = self.data[first_idx].copy()

                # initialize min_distances with infinity
                min_distances = np.full(self.n_samples, np.inf)

                # iterate over remaining k-1 centroids
                for i in range(1, self.k):
                    debug and print('iteration', i)

                    # calculate squared distance of each point to closest centroid (only calculate for new centroids)
                    new_distances = np.sum((self.data - self.centroids[i-1])**2, axis=1)
                    np.minimum(min_distances, new_distances, out=min_distances)

                    # probabilities are given by the normalized distance squared
                    probs = min_distances / np.sum(min_distances)
                    debug and print('probs:', probs)

                    # choose next centroid randomly based on cumulated probabilities
                    candidate_j = np.random.choice(self.n_samples, size=trials, p=probs)
                    debug and print('candidate_j:', candidate_j)

                    # calculate the cost of each trial
                    best_cost = np.inf
                    best_idx = -1

                    # costs = np.zeros(trials)
                    for t in range(trials):
                        temp_centroids = self.centroids[:i].copy()
                        temp_centroids = np.vstack([temp_centroids, self.data[t]])

                        distances = cdist(self.data, temp_centroids, metric='sqeuclidean')
                        assignments = np.argmin(distances, axis=1)

                        # calculate cost
                        cost = 0
                        for j in range(i+1):
                            mask = assignments == j
                            if np.any(mask):
                                cost += np.sum(distances[mask, j])

                        if cost < best_cost:
                            best_cost = cost
                            best_idx = t
                    
                    self.centroids[i] = self.data[best_idx].copy()

                    debug and print('centroids:\n', self.centroids)

                self._distance_cache = cdist(self.data, self.centroids, metric='sqeuclidean')
                self.y_pred = np.argmin(self._distance_cache, axis=1)


    def _lloyd(self, debug=0):
        """
        Lloyd's algorithm for k-means clustering.
        """

        debug and print('\nRunning Lloyd\'s algorithm...')
        old_y_pred = np.zeros(self.n_samples, dtype=int)

        while True:
            
            debug and print('New iteration')
            self.iterations += 1
            np.copyto(old_y_pred, self.y_pred)

            # move centroids to the mean of their cluster
            new_centroids = self._move_centroids(None, debug > 1)
            self.centroids = new_centroids

            # assign each data point to the closest centroid
            self.y_pred = self._assign_clusters(debug=debug>1)
            debug and print('y_pred:', self.y_pred)

            # check for convergence
            if np.array_equal(old_y_pred, self.y_pred):
                break


    def _extended_hartigan(self, always_safe=False, binary_hartigan=False, debug=0):
        """
        Extended Hartigan algorithm for k-means clustering (unsafe+safe, always safe or binary mode).
        """

        debug and print('\nRunning Extended Hartigan algorithm...')

        # move centroids to the mean of their cluster
        self.centroids = self._move_centroids(None, debug > 1)

        while True:
            self.iterations += 1

            # start with unsafe mode    
            safe_mode = False

            # create an empty dictionary of new candidates
            candidates = {}

            for datapoint_id in range(self.n_samples):
                debug and print('\ndatapoint_id:', datapoint_id)

                candidates = self._find_candidates(datapoint_id, candidates, debug)
                    
            debug and print('\ncandidates:', candidates)
            
            # break at convergence
            if not candidates:      ## [] -> False
                debug and print('no more candidates')
                break

            # proceed in unsafe mode
            if not safe_mode and not always_safe and not binary_hartigan:
                debug and print('\nentered in UNSAFE mode')

                # store current state for possible rollback
                rollback = self.y_pred.copy()

                # calculate original cost
                original_cost = self._tot_cluster_cost(self.centroids, self.y_pred, debug > 1)
                debug and print('original_cost:', original_cost)

                # accept all candidates
                new_cost, new_centroids = self._accept_candidates(candidates, debug > 1)
                debug and print('new cost:', new_cost)

                if new_cost >= original_cost:
                    # new clustering is more expensive, proceed in safe mode
                    safe_mode = True
                    self.y_pred = rollback

            # start new condition since safe mode can be entered from unsafe mode
            if (safe_mode or always_safe) and not binary_hartigan:
                debug and print('\nentered in SAFE mode')
                self.safe_iterations += 1

                # track unchanged clusters and sort candidates by delta cost (most negative first)
                unchanged_clusters = list(range(self.k))
                sorted_candidates = sorted(candidates.items(), key=lambda e: e[1][1], reverse=True)
                debug and print('\nsorted_candidates:', sorted_candidates)

                for datapoint_id, [delta_cost, current_centroid_id, new_centroid_id] in sorted_candidates:

                    # if both clusters are still unchanged, accept the candidate
                    if current_centroid_id in unchanged_clusters and new_centroid_id in unchanged_clusters:
                        debug and print(f'candidate {datapoint_id} moved from {current_centroid_id} to {new_centroid_id}')
                        self.y_pred[datapoint_id] = new_centroid_id
                        unchanged_clusters.remove(current_centroid_id)
                        unchanged_clusters.remove(new_centroid_id)

                    # if we cannot operate on any more clusters, break
                    if not unchanged_clusters:
                        break

                new_centroids = self._move_centroids(None, debug > 1)

            # proceed in binary-hartigan if needed
            elif binary_hartigan:
                debug and print('\nentered in BINARY mode')

                # store current state for possible rollback
                rollback = self.y_pred.copy()

                # calculate original cost
                original_cost = self._tot_cluster_cost(self.centroids, self.y_pred, debug > 1)
                debug and print('original_cost:', original_cost)

                candidates_partition = [candidates]
                no_edit = True
                while no_edit:
                    debug and print('candidates_partition:', candidates_partition)

                    for part in candidates_partition:
                        # "binary" split
                        candidates_items = list(part.items())
                        half = len(candidates)//2

                        part_1 = dict(candidates_items[:half])                        
                        debug and print('part_1:', part_1)
                        new_cost, new_centroids = self._accept_candidates(part_1, debug > 1)
                        debug and print('new_cost trying part_1:', new_cost)

                        if new_cost >= original_cost:
                            # new clustering accepting part_1 is more expensive
                            # rollback and try with part_2
                            self.y_pred = rollback

                            part_2 = dict(candidates_items[half:])
                            debug and print('part_2:', part_2)
                            new_cost, new_centroids = self._accept_candidates(part_2, debug > 1)
                            debug and print('new_cost trying part_2:', new_cost)
                            
                            if new_cost >= original_cost:
                                # new clustering accepting part_2 is more expensive
                                # rollback and proceed with "binary" split
                                self.y_pred = rollback
                            else:
                                no_edit = False
                                break
                        else:
                            no_edit = False
                            break

                        # if no break was encountered, proceed with "binary" split
                        candidates_partition = [part_1, part_2]

            self.centroids = new_centroids

    def _hartigan(self, debug=0):
        """
        Hartigan algorithm for k-means clustering.
        """
        debug and print('\nRunning Hartigan algorithm...')

        # TODO: correct?
        self.centroids = self._move_centroids(None, debug > 1)

        edit = True
        while edit:
            self.iterations += 1

            edit =  False
            for datapoint_id in range(self.n_samples):
                debug and print('\ndatapoint_id:', datapoint_id)

                candidate = self._find_candidates(datapoint_id, {}, debug)
                debug and print('candidate:', candidate)

                if candidate:
                    new_cost, new_centroids = self._accept_candidates(candidate, debug > 1)
                    self.centroids = new_centroids
                    edit = True
                    # the code continues with the next datapoint instead than starting from the first one again
                    # TODO: better to start from the first one again? can depend on the index of the datapoint we edit first?

    def _move_centroids(self, move_just = None, debug=0):
        """
        Move the centroids to the mean of their cluster.
        """

        debug and print('\n  moving centroids...')
        debug and print('  | y_pred:', self.y_pred)
        debug and print('  | data:\n', self.data)
        debug and print('  | centroids_before:\n', self.centroids)

        centroids = np.copy(self.centroids)
        
        move = move_just if move_just is not None else range(self.k)
        debug and print('  | move:', move)
        
        for centroid_id in move:
            
            mask = self.y_pred == centroid_id
            
            # if centroid has no points assigned to it, reassign it randomly
            if np.any(mask):
                centroids[centroid_id] = np.mean(self.data[mask], axis=0)
            else:
                debug and print(f"  Centroid {centroid_id} is empty. Reassigning.")
                new_centroid_id = np.random.choice(self.n_samples)
                centroids[centroid_id] = self.data[new_centroid_id]
                self.y_pred[new_centroid_id] = centroid_id

        debug and print('  centroids_after:\n', centroids)

        return centroids


    def _assign_clusters(self, specific_centroids=None, debug=0):
        """
        Assign each data point to the closest centroid.

        Parameters
        ----------
        specific_centroids : np.ndarray
            array of shape (k', d) with specific centroids (of a specific number k') to use for the assignment

        Returns
        -------
        np.ndarray
            Array of length n with cluster assignments for each sample
        """
        
        if specific_centroids is None:
            specific_centroids = self.centroids
        
        self._distance_cache = cdist(self.data, self.centroids, metric='sqeuclidean')
        y_pred = np.argmin(self._distance_cache, axis=1)
        self.norm_calculations += specific_centroids.shape[0] / (self.k*self.n_samples)

        debug and print('y_pred:', y_pred)

        return y_pred


    def _tot_cluster_cost(self, centroids, points_ids, debug=0):
        """
        Compute the overall cost of clustering

        Parameters
        ----------
        centroids : np.ndarray
            Array of shape ('k, d) with cluster centroids
        points_ids : np.ndarray
            Array of length n with cluster assignments for each sample

        Returns
        -------
        float
            Total cost of clustering
        """
        
        debug and print('\n  calculating _tot_cluster_cost')
        
        total_cost = 0

        for centroid_id in range(centroids.shape[0]):
            mask = points_ids == centroid_id
            points_in_cluster = self.data[mask]

            cluster_items = np.where(points_ids == centroid_id)[0]

            if len(points_in_cluster) > 0:
                # Vectorized computation
                cost = np.sum(np.sum((points_in_cluster - centroids[centroid_id])**2, axis=1))
                total_cost += cost
                self.norm_calculations += len(cluster_items) / (self.k*self.n_samples)

            debug and print('  | centroid_id:', centroid_id)
            debug and print('  | centroid:', centroids[centroid_id])
            debug and print('  | cluster_items:', cluster_items)
            debug and print('  | cost:', cost)
        
        debug and print(f'  total_cost: {total_cost}')
        
        return total_cost


    def _find_candidates(self, datapoint_id, candidates, debug=0):
        """
        Find candidates for reassignment of a single datapoint.
        """

        # calculate sizes of clusters and cost of current assignment
        cluster_sizes = np.bincount(self.y_pred, minlength=self.k)

        current_centroid_id = self.y_pred[datapoint_id]
        current_size = cluster_sizes[current_centroid_id]
        
        # if current clsuter has only one point, we cannot move it
        if current_size <= 1:
            return candidates
        
        prefactor = current_size / (current_size - 1)
        current_cost = prefactor * np.sum((self.data[datapoint_id] - self.centroids[current_centroid_id])**2)
        self.norm_calculations += 1/(self.k*self.n_samples)

        debug and print('current_cost:', current_cost)

        # if current_cost is 0, delta_cost will always be positive
        if current_cost == 0:
            return candidates
        
        # Vectorized computation for all centroids except the current one
        mask = np.arange(self.k) != current_centroid_id
        cluster_sizes_masked = cluster_sizes[mask]
        valid_centroid_ids = np.where(mask)[0]

        # Compute delta costs
        delta_costs = (cluster_sizes_masked / (cluster_sizes_masked + 1)) * np.linalg.norm(self.data[datapoint_id] - self.centroids[valid_centroid_ids], axis=1)**2 - current_cost
        self.norm_calculations += (len(cluster_sizes_masked)) / (self.k*self.n_samples)

        # Find the best centroid (most negative delta_cost)
        best_idx = np.argmin(delta_costs)
        best_delta_cost = delta_costs[best_idx]

        if best_delta_cost < 0:
            best_centroid_id = valid_centroid_ids[best_idx]
            candidates[datapoint_id] = [best_delta_cost, current_centroid_id, best_centroid_id]
        
        return candidates


    def _accept_candidates(self, candidates, debug=0):
        """
        Accepts all candidates passed as argument and calculates new total cluster cost.
        """
        # accept all candidates
        used_centroids = set()

        for datapoint_id, [_, current_centroid_id, new_centroid_id] in candidates.items():

            debug and print('candidate:', datapoint_id)

            used_centroids.add(current_centroid_id)
            used_centroids.add(new_centroid_id)

            debug and print('y_pred before:', self.y_pred)

            # update closest_points_ids assigning datapoint to new_centroid_id
            self.y_pred[datapoint_id] = new_centroid_id
            debug and print('y_pred after:', self.y_pred)

        new_centroids = self._move_centroids(move_just=used_centroids, debug = debug)

        return self._tot_cluster_cost(new_centroids, self.y_pred, debug), new_centroids