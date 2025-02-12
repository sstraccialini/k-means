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
                 init : Literal['random', 'random-data', 'k-means++', 'maximin'] = 'random',
                 seed : Union[int, None] = None):
        """
        Initialize the KMeans object.

        Parameters
        ----------
        algorithm : {'lloyd', 'extended-hartigan', 'safe-hartigan', 'hartigan', 'binary-hartigan'}
            Algorithm to use. Either 'lloyd' or 'extended-hartigan' or 'safe-hartigan' or 'hartigan' or 'binary-hartigan'

        init : {'random', 'random-data', 'k-means++', 'maximin'}
            Initialization method. Either 'random' or 'random-data' or 'k-means++' or 'maximin'

        seed : int
            Seed for random generator
        """

        assert algorithm in ['lloyd', 'extended-hartigan', 'safe-hartigan', 'hartigan', 'binary-hartigan'], "algorithm must be either 'lloyd', 'extended-hartigan', 'safe-hartigan', 'hartigan' or 'binary-hartigan'"
        assert init in ['random', 'random-data', 'k-means++', 'maximin'], "init must be either 'random', 'random-data', 'k-means++' or 'maximin'"
        assert seed is None or isinstance(seed, int), "seed must be an int or None"

        self.algorithm = algorithm
        self.init = init
        self.seed = seed

        self.data = None
        self.k = None
        self.centroids = None
        self.y_pred = None

        self.safe_iterations = 0


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
        
        print('final centroids:\n', self.centroids)
        print('final y_pred:', self.y_pred)


    def _init_centroids(self, debug=0):
        """
        Initialize the centroids.
        """

        if self.init == 'random':

            # choose k random data points as initial centroids
            idx = np.random.choice(self.data.shape[0], self.k, replace=False)
            self.centroids = self.data[idx]
            self.y_pred = self._assign_clusters(debug > 1)

        elif self.init == 'random-data':

            # assign each data point to a random cluster
            clusters = np.random.choice(self.k, self.data.shape[0])

            # check that at least one point is assigned to each cluster
            while len(set(clusters)) < self.k:
                clusters = np.random.choice(self.k, self.data.shape[0])
            self.y_pred = clusters
            self.centroids = self._move_centroids(None, debug > 1)

        elif self.init == 'k-means++':

            # choose first centroid randomly
            centroids = np.zeros((self.k, self.data.shape[1]))
            centroids[0] = self.data[np.random.choice(self.data.shape[0], 1, replace=False)[0]]
            debug and print('centroids:\n', centroids)

            # iterate over remaining k-1 centroids
            for i in range(1, self.k):
                debug and print('iteration', i)

                # calculate squared distance of each point to closest centroid
                dist = np.array([min([np.linalg.norm(c-x)**2 for c in centroids[:i]]) for x in self.data])

                # probabilities are given by the normalized distance squared
                probs = dist / dist.sum()
                debug and print('probs:', probs)

                # choose next centroid randomly based on cumulated probabilities
                j = np.random.choice(len(self.data), p=probs)

                centroids[i] = self.data[j]
                debug and print('centroids:\n', centroids)

            self.centroids = centroids
            self.y_pred = self._assign_clusters(debug > 1)

        elif self.init == 'maximin':

            # choose first centroid randomly
            centroids = np.zeros((self.k, self.data.shape[1]))
            centroids[0] = self.data[np.random.choice(self.data.shape[0], 1, replace=False)[0]]
            debug and print('centroids:\n', centroids)

            # iterate over remaining k-1 centroids
            for i in range(1, self.k):
                debug and print('iteration', i)

                # calculate squared distance of each point to closest centroid
                dist = np.array([min([np.linalg.norm(c-x)**2 for c in centroids[:i]]) for x in self.data])

                # choose next centroid as the point with the maximum distance to the closest centroid
                centroids[i] = self.data[np.argmax(dist)]
                debug and print('centroids:\n', centroids)

            self.centroids = centroids
            self.y_pred = self._assign_clusters(debug > 1)


    def _lloyd(self, debug=0):
        """
        Lloyd's algorithm for k-means clustering.
        """

        debug and print('\nRunning Lloyd\'s algorithm...')

        while True:
            
            debug and print('New iteration')

            # move centroids to the mean of their cluster
            new_centroids = self._move_centroids(None, debug > 1)
            self.centroids = new_centroids

            # assign each data point to the closest centroid
            old_y_pred = self.y_pred
            self.y_pred = self._assign_clusters(debug > 1)
            debug and print('y_pred:', self.y_pred)

            # check for convergence
            if np.array_equal(old_y_pred, self.y_pred):
                break



    def _extended_hartigan(self, always_safe=False, binary_hartigan=False, debug=0):
        """
        Extended Hartigan algorithm for k-means clustering (unsafe+safe, always safe or binary mode).
        """

        debug and print('\nRunning Extended Hartigan algorithm...')

        # TODO: correct?
        self.centroids = self._move_centroids(None, debug > 1)

        while True:
            # start with unsafe mode    
            safe_mode = False

            # create an empty dictionary of new candidates
            candidates = {}

            for datapoint_id in range(len(self.data)):
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

                unchanged_clusters = list(range(self.k))
                debug and print('\ncandidates:', sorted(candidates.items(), key=lambda e: e[1][1], reverse=True))
                for _, [delta_cost, current_centroid_id, new_centroid_id] in sorted(candidates.items(), key=lambda e: e[1][1], reverse=True):

                    # if both clusters are still unchanged, accept the candidate
                    if current_centroid_id in unchanged_clusters and new_centroid_id in unchanged_clusters:
                        debug and print(f'candidate {_} moved from {current_centroid_id} to {new_centroid_id}')
                        self.y_pred[_] = new_centroid_id
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
            edit =  False
            for datapoint_id in range(len(self.data)):
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
            
            cluster_points = self.data[self.y_pred == centroid_id]
            
            # if centroid has no points assigned to it, reassign it randomly
            if len(cluster_points) == 0: # TODO: makes sense?
                debug and print(f"  Centroid {centroid_id} is empty. Reassigning.")
                new_centroid_id = np.random.choice(len(self.data))
                centroids[centroid_id] = self.data[new_centroid_id]
                self.y_pred[new_centroid_id] = centroid_id
            else:
                centroids[centroid_id] = np.mean(cluster_points, axis=0)

        debug and print('  centroids_after:\n', centroids)

        return centroids


    def _assign_clusters(self, debug=0):
        """
        Assign each data point to the closest centroid.
        """
        
        distances = cdist(self.data, self.centroids, metric='sqeuclidean')  # Squared Euclidean distance
        y_pred = np.argmin(distances, axis=1)

        debug and print('y_pred:', y_pred)

        return y_pred


    def _delta_cost(self, cost, datapoint_id, centroid_id):
        """
        Compute the change in cost if datapoint is reassigned to centroid_id
        """

        cluster_size = np.where(self.y_pred == centroid_id)[0].shape[0]
        prefactor = cluster_size / (cluster_size + 1)

        # cost of new assignment
        new_cost = prefactor * np.linalg.norm(self.data[datapoint_id] - self.centroids[centroid_id])**2

        return new_cost - cost


    def _tot_cluster_cost(self, centroids, points_ids, debug=0):
        """
        Compute the overall cost of clustering
        """
        
        debug and print('\n  calculating _tot_cluster_cost')
        
        partial_sum = []
        for centroid_id in range(centroids.shape[0]):
            cluster_items = np.where(points_ids == centroid_id)[0]
            partial_sum.append(np.sum(np.square(self.data[cluster_items] - self.centroids[centroid_id])))

            debug and print('  | centroid_id:', centroid_id)
            debug and print('  | centroid:', centroids[centroid_id])
            debug and print('  | cluster_items:', cluster_items)
            debug and print('  | partial_sum:', partial_sum)
        
        debug and print('  partial_sum:', np.sum(partial_sum))
        debug and print('  _tot_cluster_cost:', np.sum(partial_sum))
        
        return np.sum(partial_sum)


    def _find_candidates(self, datapoint_id, candidates, debug=0):
        """
        Find candidates for reassignment of a single datapoint.
        """

        # calculate cost of current assignment which remains invariant
        current_centroid_id = self.y_pred[datapoint_id]
        cluster_size = np.where(self.y_pred == current_centroid_id)[0].shape[0]
        prefactor = cluster_size / (cluster_size - 1) if cluster_size > 1 else 0

        current_cost = prefactor * np.linalg.norm(self.data[datapoint_id] - self.centroids[current_centroid_id])**2
        debug and print('current_cost:', current_cost)

        # if current_cost is 0, delta_cost will always be positive
        if current_cost == 0:
            return candidates

        # iterate only on possible new centroid assignments
        for centroid_id in set(self.y_pred) - {current_centroid_id}:
            delta_cost = self._delta_cost(current_cost, datapoint_id, centroid_id)
            debug and print(f'delta_cost for datapoint {datapoint_id} from centroid {current_centroid_id} to centroid {centroid_id}:', delta_cost)

            # datapoint is a candidate if it reduces the cost
            # if more reassignments reduce the cost, the best one is stored (the one producing the most negfative delta_cost)
            if delta_cost < 0 and (candidates.get(datapoint_id) is None or delta_cost < candidates[datapoint_id][0]):
                candidates[datapoint_id] = [delta_cost, current_centroid_id, centroid_id]

        return candidates


    def _accept_candidates(self, candidates, debug=0):
        """
        Accepts all candidates passed as argument and calculates new total cluster cost.
        """
        # accept all candidates
        used_centroids = set()
        for candidate in candidates.keys():
            debug and print('candidate:', candidate)

            [delta_cost, current_centroid_id, new_centroid_id] = candidates[candidate]


            used_centroids.add(current_centroid_id)
            used_centroids.add(new_centroid_id)

            debug and print('y_pred before:', self.y_pred)

            # update closest_points_ids assigning datapoint to new_centroid_id
            self.y_pred[candidate] = new_centroid_id
            debug and print('y_pred after:', self.y_pred)

        new_centroids = self._move_centroids(move_just=used_centroids, debug = debug)

        return self._tot_cluster_cost(new_centroids, self.y_pred, debug), new_centroids


# debug
a = np.array([[1,1],[2,1],[1,2],[2,3],[3,2],[9,9],[10,8],[10,10],[11,9]])

with Profile() as profile:

    for i in range(1):
        a1 = pd.read_table('data/A-Sets/a3.txt', header=None, sep='   ', engine='python').to_numpy()
        kmeans = KMeans(algorithm='hartigan', init='random', seed=i)
        kmeans.fit(a1, 20, debug=0)
        with open('profiling/results.txt', 'w') as f:
            print(kmeans.centroids, file=f)
            print(kmeans.y_pred, file=f)
    (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)
        .print_stats()
    )

with open('profiling/profiling.txt', 'w') as f:
    Stats(profile, stream=f).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()