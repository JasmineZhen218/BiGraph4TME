import numpy as np
import networkx as nx


class Population_Graph:
    def __init__(self, k=20, resolution=1.0, size_smallest_cluster=10, seed=1):
        self.k = k # number of nearest neighbors in the population graph
        self.resolution = resolution # resolution parameter for community detection
        self.seed = seed # random seed
        self.size_smallest_cluster = size_smallest_cluster # the size of the smallest cluster
        print("Initialized Population_Graph, k = ", k, ", resolution = ", resolution, ", size_smallest_cluster = ", size_smallest_cluster, ", seed = ", seed)


    def generate(self, Similarity_matrix, Patient_ids):
        """
        Generate the population graph.
        Parameters
        ----------
        Similarity_matrix : numpy array (n_patients, n_patients)
            The similarity matrix between patients.
        Patient_ids : list of str
            The patient ids.
        Returns
        -------
        Population_graph : networkx graph
            The population graph.
        """
        Similarity_matrix_ = Similarity_matrix # make a copy
        np.fill_diagonal(Similarity_matrix_, 0)  # remove self-similarity
        
        for i in range(Similarity_matrix_.shape[0]):
            idx = np.argsort(Similarity_matrix_[i])[
                ::-1
            ]  # sort the similarity in descending order
            Similarity_matrix_[i, idx[self.k :]] = (
                0  # remove edges that are not in the k nearest neighbors
            )
            adj_1 = np.maximum(
                Similarity_matrix_, Similarity_matrix_.transpose()
            )  # make it symmetric
            adj_2 = np.zeros_like(adj_1) # IoU matrix
            for i in range(Similarity_matrix_.shape[0]): 
                for j in range(Similarity_matrix_.shape[0]):
                    neighbor_i = np.where(adj_1[i, :] > 0)[0]
                    neighbor_j = np.where(adj_1[j, :] > 0)[0]
                    IoU = len(set(neighbor_i).intersection(set(neighbor_j))) / len(
                        set(neighbor_i).union(set(neighbor_j)) # calculate the IoU
                    )
                    adj_2[i, j] = IoU # fill in the IoU matrix
            np.fill_diagonal(adj_2, 0) # remove self-similarity
            G_population = nx.from_numpy_array(adj_2) # create the population graph
            Patient_ids_dict = {i: Patient_ids[i] for i in range(len(Patient_ids))} # create a dictionary for patient ids
            nx.set_node_attributes(G_population, Patient_ids_dict, "patientID") # set the patient ids as node attributes
            return G_population

    def community_detection(self, G_population):
        """
        Community detection on the population graph.
        Parameters
        ----------
        G_population : networkx graph
            The population graph.
        Returns
        -------
        Patient_subgroups : dict
            The patient subgroups. The key is the subgroup id, and the value is a list of patient ids.
        """
        Communities = nx.algorithms.community.louvain_communities(
            G_population, weight="weight", resolution=self.resolution, seed=1
        ) # community detection using Louvain method
        Communities = [list(c) for c in Communities if len(c) > self.size_smallest_cluster] # remove small clusters
        Communities = sorted(Communities, key=lambda x: len(x), reverse=True) # sort the clusters by size
        Patient_subgroups = {i: [] for i in range(len(Communities))} # create a dictionary for patient subgroups
        for i, c in enumerate(Communities):
            Patient_subgroups[i] = [G_population.nodes[n]["patientID"] for n in c] # fill in the patient subgroups
        return Patient_subgroups

