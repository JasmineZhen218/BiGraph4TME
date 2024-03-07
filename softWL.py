# Implementation of Soft WL subtree kernel 
# -- a relaxation of the conventional Weisfeiler-Lehman subtree kernel
import numpy as np
import phenograph

class SoftWL():
    """Calculate the Soft WL subtree kernel"""
    def __init__(self, n_iter=0,  n_jobs = -1, k = 100, normalize=True):
        self.n_iter = n_iter # number of iterations of graph convolution
        self.n_job = n_jobs # number of jobs for parallel computation
        self.k = k # number of neighbors for phenograph clustering
        self.normalize = normalize # whether to normalize the kernel matrix
    def graph_convolution(self, adj, x):
        pass
    def cluster_subtrees(self, X):
        cluster_identities, _, _ = phenograph.cluster(X, n_jobs=self.n_job, k=self.k)
        return cluster_identities
    def compute_cluster_centroids(X, Cluster_identities):
        pass
    def compute_histogram(self, X):
        pass
    def closest_cluster_mapping(self, X, Signatures):
        pass

    def discover_patterns(self, X):
        """ Given a set of cellular graphs --> generate subtrees --> Discover TME patterns by clustering subtrees
        Parameters
        ----------
        X : list
            Each element is a tuple: (adj, x)
            adj is the adjacency matrix (N x N) while x is the node label/attribute matrix (N x d).
                N: number of nodes in a graph
                d: dimension of node features (e.g., one-hot encoding of cell type)
        Returns
        -------
        X_prime: list
            Each element is a tuple: (adj, x)
            adj is the adjacency matrix (N x N) while x  is the resultant pattern id (N x 1).
                N: number of nodes in a graph
        self.Signatures: numpy array, shape = [n_patterns, n_features]
        """
        Subtree_features = [] # list of graph convolution results
        N_nodes = [] # list of number of nodes in each graph
        for i, (adj, x) in enumerate(X): # iterate through the graphs
            subtree_feature = self.graph_convolution(adj, x) # subtree_feature for each graph
            Subtree_features.append(subtree_feature) # append the subtree feature
            N_nodes.append(subtree_feature.shape[0]) # append the number of nodes
        Subtree_features = np.concatenate(Subtree_features, axis=0) # concatenate all subtree features
        Pattern_ids = self.cluster_subtrees(Subtree_features) # cluster the subtree features
        Signatures = self.compute_cluster_centroids(Subtree_features, Pattern_ids)  # compute the cluster centroids --> signature of each TME pattern
        X_prime = [] # list of graphs with pattern ids
        start = 0 # start index of the pattern ids
        for i, n in enumerate(N_nodes): # iterate through the graphs
            end = start + n # end index of the pattern ids
            X_prime.append((X[i][0], Pattern_ids[start:end])) # append the graph with pattern ids
            start = end # update the start index
        self.Signatures = Signatures # store the signatures
        self.num_patterns = Signatures.shape[0] # store the number of patterns
        self.X = X # store the input graphs
        self.X_prime = X_prime # store the graphs with pattern ids
        return X_prime
    
    def estimate_patterns(self, X):
        """ Given a set of cellular graphs --> generate subtrees --> estimate the pattern belongingness of each subtree 
        Parameters
        ----------
        X : list
            Each element is a tuple: (adj, x)
            adj is the adjacency matrix (N x N) while x is the node label/attribute matrix (N x d).
                N: number of nodes in a graph
                d: dimension of node features (e.g., one-hot encoding of cell type)
        Returns
        -------
        X_prime: list
            Each element is a tuple: (adj, x)
            adj is the adjacency matrix (N x N) while x  is the resultant pattern id (N x 1).
                N: number of nodes in a graph
        """
        Subtree_features = []
        N_nodes = []
        for i, (adj, x) in enumerate(X):
            subtree_feature = self.graph_convolution(adj, x)
            Subtree_features.append(subtree_feature)
            N_nodes.append(subtree_feature.shape[0])
        Subtree_features = np.concatenate(Subtree_features, axis=0)
        Pattern_ids = self.closest_cluster_mapping(Subtree_features, self.Signatures)
        X_prime = []
        start = 0 # start index of the pattern ids
        for i, n in enumerate(N_nodes): # iterate through the graphs
            end = start + n # end index of the pattern ids
            X_prime.append((X[i][0], Pattern_ids[start:end])) # append the graph with pattern ids
            start = end # update the start index
        return X_prime


    def fit_transform(self, X):
        """
        Given a set of cellular graphs, generate the TME patterns and compute the signature of each pattern (fit), and then calculate the kernel matrix (transform).
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
        K : numpy array, shape = [len(X), len(X)]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        X_prime = self.discover_patterns(X) # discover the TME patterns
        Histograms = self.compute_histograms(X_prime) # compute the histograms
        self.Histograms = Histograms # store the histograms
        # Initialize the kernel matrix
        n = len(X)
        K = np.zeros((n, n))
        # Iterate through the graphs
        for i, histogram_i in enumerate(self.Histograms):
            for j, histogram_j in enumerate(self.Histograms):
                K[i, j] = np.inner(histogram_i, histogram_j) # calculate the kernel matrix: inner product of histograms
        if self.normalize: 
            K = K / np.sqrt(np.outer(np.diag(K), np.diag(K))) # normalize the kernel matrix
        return K
    
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
        X_prime = self.estimate_patterns(X) # estimate the pattern belongingness of each subtree
        Histograms = self.compute_histograms(X_prime) # compute the histograms
        Histograms_fitted = self.Histograms # retrieve the histograms of the fitted graphs
        Histograms_new = Histograms # retrieve the histograms of the new graphs
        n_fitted = len(Histograms_fitted)  
        n_new = len(Histograms_new) 
        K = np.zeros((n_fitted, n_new)) # Initialize the kernel matrix
        for i, histogram_i in enumerate(Histograms_fitted): # Iterate through the fitted graphs
            for j, histogram_j in enumerate(Histograms_new): # Iterate through the new graphs
                K[i, j] = np.inner(histogram_i, histogram_j) # calculate the kernel matrix: inner product of histograms
        if self.normalize: 
            K = K / np.sqrt(np.outer(np.diag(K), np.diag(K))) # normalize the kernel matrix
        return K


