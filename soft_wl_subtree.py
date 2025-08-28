# Implementation of Soft WL subtree kernel
# -- a relaxation of the conventional Weisfeiler-Lehman subtree kernel
import numpy as np
import phenograph
from sklearn.neighbors import NearestNeighbors


class Soft_WL_Subtree(object):
    """Calculate the Soft WL subtree kernel"""

    def __init__(self, n_iter=0, n_jobs=-1, k=100, normalize=True):
        self.n_iter = n_iter  # number of iterations of graph convolution
        self.n_job = n_jobs  # number of jobs for parallel computation
        self.k = k  # number of neighbors for phenograph clustering
        self.normalize = normalize  # whether to normalize the kernel matrix
        print(
            "Initialize SoftWL: n_iter={}, n_jobs={}, k={}, normalize={}".format(
                self.n_iter, self.n_job, self.k, self.normalize
            )
        )
        self.Signatures = None  # initialize the signatures of the TME patterns
        self.Histograms = None  # initialize the histograms of the graphs
        self.num_patterns = None  # initialize the number of patterns
        self.X = None  # initialize the input graphs
        self.X_prime = None  # initialize the graphs with pattern ids
        self.Similarity_matrix = None  # initialize the kernel matrix

    def graph_convolution(self, adj, x):
        """
        graph convolution
        Parameters
        ----------
        adj: numpy array, shape = [n_samples, n_samples]
            adjacency matrix
        x: numpy array, shape = [n_samples, n_features]
            node label/attribute matrix
        Returns
        -------
        x: numpy array, shape = [n_samples, n_features]
            node label/attribute matrix after graph convolution
        """
        np.fill_diagonal(adj, 1)  # set the diagonal of the adjacency matrix to 1
        for i in range(self.n_iter):  # iterate through the number of iterations
            x = np.dot(adj, x)
        return x

    def cluster_subtrees(self, X):
        # call "export OMP_NUM_THREADS=1" before running to avoid "Too many memory regions" error with Dask
        """
        Cluster the subtrees
        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
        Returns
        -------
        cluster_identities: numpy array, shape = [n_samples]
        """
        cluster_identities, _, _ = phenograph.cluster(X, n_jobs=self.n_job, k=self.k)
        return cluster_identities

    def compute_cluster_centroids(self, X, Cluster_identities):
        """
        Compute the cluster centroids
        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
        Cluster_identities: numpy array, shape = [n_samples]
        Returns
        -------
        Signatures: numpy array, shape = [n_clusters, n_features]
        """
        n_clusters = len(np.unique(Cluster_identities))
        Signatures = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            Signatures[i] = np.mean(X[Cluster_identities == i], axis=0)
        return Signatures

    def compute_histograms(self, X):
        """
        Compute the histogram of the patterns
        Parameters
        ----------
        X: list
            Each element is a tuple: (patient_id, adj, subtree_features, x)
            adj is the adjacency matrix (N x N) while x is the pattern id matrix (N).
                N: number of nodes in a graph
        Returns
        -------
        Histograms: list
            Each element is a numpy array, shape = [self.n_patterns]
        """
        Histograms = []
        for i, (_, _,_, x) in enumerate(X):
            histogram = np.zeros(self.num_patterns)
            for j in range(self.num_patterns):
                histogram[j] = np.sum(x == j)
            Histograms.append(histogram)
        return Histograms

    def closest_cluster_mapping(self, X, Signatures):
        """
        Given a set of subtrees, find the closest cluster
        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
        Signatures: numpy array, shape = [n_clusters, n_features]
        Returns
        -------
        Pattern_ids_hat: numpy array, shape = [n_samples]
        """
        neigh = NearestNeighbors(n_neighbors=1, radius=1)
        neigh.fit(Signatures)
        distances, indices = neigh.kneighbors(X)
        Pattern_ids_hat = indices.flatten()
        return Pattern_ids_hat

    def discover_patterns(self, X, sample_frac=0.8, random_state=None):
        """
        Given a set of cellular graphs --> generate subtrees --> (optionally) sample nodes
        --> Discover TME patterns by clustering subtrees.

        Parameters
        ----------
        X : list
            Each element is a tuple: (patient_id, adj, x)
            patient_id: str
            adj: numpy array, shape = [n_nodes, n_nodes]  (adjacency matrix)
            x:   numpy array, shape = [n_nodes, n_features] (node attributes)
        sample_frac : float
            Fraction of nodes to keep per patient for clustering. Default 0.8.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        X_prime: list
            Each element is a tuple: (patient_id, adj_sub, feat_sub, pattern_ids_sub, kept_idx)
            adj_sub: adjacency of sampled nodes (shape = [k_i, k_i])
            feat_sub: subtree features of sampled nodes (shape = [k_i, d'])
            pattern_ids_sub: length k_i
            kept_idx: indices (into the original node order) that were kept
        Signatures: numpy array, shape = [n_patterns, d']
            Cluster centroids.
        """
        rng = np.random.default_rng(random_state)

        print(
            "Discovering TME patterns from {} graphs, median number of nodes is {}, node feature matrix dimension is {}".format(
                len(X), np.median([x[1].shape[0] for x in X]), X[0][2].shape
            )
        )

        Subtree_features = []   # concatenated sampled subtree features across all graphs
        N_nodes = []            # sampled count per graph (k_i)
        kept_indices_per_graph = []  # store which nodes we kept per graph for traceability

        print("\t 1) Graph Convolution")
        per_graph_features = []  # hold per-graph sampled features to avoid recomputing later

        for i, (patient_id, adj, x) in enumerate(X):
            # Compute per-node subtree features for this graph
            subtree_feature = self.graph_convolution(adj, x)  # shape [n_i, d']
            n_i = subtree_feature.shape[0]

            # Sample 80% (at least 1 node)
            k_i = max(1, int(np.ceil(sample_frac * n_i)))
            kept_idx = rng.choice(n_i, size=k_i, replace=False)
            kept_idx.sort()  # keep order stable (optional)

            # Slice features (used for clustering) and remember counts/indices
            feat_sub = subtree_feature[kept_idx, :]
            Subtree_features.append(feat_sub)
            N_nodes.append(k_i)
            kept_indices_per_graph.append(kept_idx)
            per_graph_features.append(feat_sub)

        # Concatenate sampled features across all graphs for clustering
        Subtree_features = np.concatenate(Subtree_features, axis=0)

        print("\t 2) Clustering {} subtrees".format(Subtree_features.shape[0]))
        Pattern_ids = self.cluster_subtrees(Subtree_features)  # length = sum_i k_i
        Signatures = self.compute_cluster_centroids(Subtree_features, Pattern_ids)

        print("\t 3) Assigning Pattern Ids to Subtrees (sampled nodes only)")
        X_prime = []
        start = 0
        for i, (patient_id, adj, x) in enumerate(X):
            k_i = N_nodes[i]
            end = start + k_i

            kept_idx = kept_indices_per_graph[i]

            # Subset adjacency to the sampled nodes (IMPORTANT so shapes match)
            adj_sub = adj[np.ix_(kept_idx, kept_idx)]

            # Use the already computed sampled subtree features for this graph
            feat_sub = per_graph_features[i]  # shape [k_i, d']

            # Pattern ids for these sampled nodes
            pattern_ids_sub = Pattern_ids[start:end]

            # Package: patient id, sub-adj, sub-features, pattern ids, and kept indices
            X_prime.append(
                (patient_id, adj_sub, feat_sub, pattern_ids_sub)
            )

            start = end

        return X_prime, Signatures

    def estimate_patterns(self, X):
        """
        Given a set of cellular graphs --> generate subtrees --> estimate the pattern belongingness of each subtree
        Parameters
        ----------
        X : list
            Each element is a tuple: (patient_id, adj, x)
            patient_id: str
            adj: numpy array, shape = [n_nodes, n_nodes]
                adjacency matrix
            x: numpy array, shape = [n_nodes, n_features]
        Returns
        -------
        X_prime: list
            Each element is a tuple: (patient_id, adj, subtree_features, pattern_ids)
            - adj is the adjacency matrix (N x N) while x  is the resultant pattern id (N x 1).
                N: number of nodes in a graph
            - pattern_ids: numpy array, shape = [n_nodes]
                pattern ids of each node
        """
        print(
            "Estimating TME patterns from {} graphs, median number of nodes is {}, node feature dimension is {}".format(
                len(X), np.median([x[1].shape[0] for x in X]), X[0][2].shape[1]
            )
        )
        neigh = NearestNeighbors(n_neighbors=1)
        #Signature_norm =  (self.Signatures - np.mean(self.Signatures, axis=0)) / np.std(self.Signatures, axis=0)
        neigh.fit(self.Signatures)
        X_prime = []
        for i, (patient_id, adj, x) in enumerate(X):
            subtree_feature = self.graph_convolution(adj, x)
            _, indices = neigh.kneighbors(subtree_feature)
            X_prime.append((patient_id, adj, subtree_feature, indices.flatten()))
        print("TME patterns estimation finished")
        return X_prime

    def fit_transform(self, X, sample_frac=0.8):
        """
        Given a set of cellular graphs, generate the TME patterns and compute the signature of each pattern (fit), and then calculate the kernel matrix (transform).
        Parameters
        ----------
        X : list of tuples, with each element being a tuple: (patient_id, adj, x)
            patient_id: str
            adj: numpy array, shape = [n_nodes, n_nodes]
                adjacency matrix
            x: numpy array, shape = [n_nodes, n_features]
                node label/attribute matrix

        Returns
        -------
        K : numpy array, shape = [len(X), len(X)]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        X_prime, Signatures = self.discover_patterns(X, sample_frac=sample_frac)  # discover the TME patterns
        self.X = X  # store the input graphs
        self.X_prime = X_prime  # store the graphs with pattern ids
        self.Signatures = Signatures  # store the signatures
        self.num_patterns = len(Signatures)  # store the number of patterns
        Histograms = self.compute_histograms(X_prime)  # compute the histograms
        self.Histograms = Histograms  # store the histograms
        # Initialize the kernel matrix
        n = len(X)
        K = np.zeros((n, n))
        # Iterate through the graphs
        for i, histogram_i in enumerate(self.Histograms):
            for j, histogram_j in enumerate(self.Histograms):
                K[i, j] = np.inner(
                    histogram_i, histogram_j
                )  # calculate the kernel matrix: inner product of histograms
        if self.normalize:
            K = K / np.sqrt(
                np.outer(np.diag(K), np.diag(K))
            )  # normalize the kernel matrix
        self.Similarity_matrix = K  # store the kernel matrix
        return K  # return the kernel matrix

    def transform(self, X):
        """Calculate the kernel matrix, between the fitted graphs and the (unseen) input graphs
        Parameters
        ----------
        X : list of tuples, with each element being a tuple: (adj, x)
            Each element must be an iterable with at two features.
            The first is the adjacency matrix (N x N) while the second is
            node label/attribute matrix (N x d).
                N: number of nodes
                d: dimension of node features (e.g., one-hot encoding of cell type)
        Returns
        -------
        K : numpy array, shape = [len(self.X), len(X)]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        X_prime = self.estimate_patterns(
            X
        )  # estimate the pattern belongingness of each subtree
        Subtree_features_new = np.concatenate(
            [x[2] for x in X_prime], axis=0
        )
        Pattern_ids_new = np.concatenate(
            [x[3] for x in X_prime], axis=0
        )
        Signatures_new = self.compute_cluster_centroids(
            Subtree_features_new, Pattern_ids_new
        ) # compute the cluster centroids --> signature of each TME pattern for new graphs
        Histograms = self.compute_histograms(X_prime)  # compute the histograms
        Histograms_fitted = (
            self.Histograms
        )  # retrieve the histograms of the fitted graphs
        Histograms_new = Histograms  # retrieve the histograms of the new graphs
        n_fitted = len(Histograms_fitted)
        n_new = len(Histograms_new)
        K = np.zeros((n_fitted, n_new))  # Initialize the kernel matrix
        for i, histogram_i in enumerate(
            Histograms_fitted
        ):  # Iterate through the fitted graphs
            for j, histogram_j in enumerate(
                Histograms_new
            ):  # Iterate through the new graphs
                K[i, j] = np.inner(
                    histogram_i, histogram_j
                )  # calculate the kernel matrix: inner product of histograms
                if self.normalize:
                    K[i, j] = K[i, j] / np.sqrt(
                        np.inner(histogram_i, histogram_i)
                        * np.inner(histogram_j, histogram_j)
                    )
                    # normalize the kernel matrix
        K_itself = np.zeros((n_new, n_new))  # Initialize the kernel matrix
        for i, histogram_i in enumerate(Histograms_new):
            for j, histogram_j in enumerate(Histograms_new):
                K_itself[i, j] = np.inner(histogram_i, histogram_j)
        if self.normalize:
            K_itself = K_itself / np.sqrt(
                np.outer(np.diag(K_itself), np.diag(K_itself))
            )
        return K, K_itself, Histograms_new, Signatures_new
