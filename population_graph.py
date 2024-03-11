import numpy as np
import networkx as nx


class Population_Graph:
    def __init__(self, k_clustering=20, k_estimate = 3, resolution=1.0, size_smallest_cluster=10, seed=1):
        self.k_clustering = k_clustering # decides the coarseness of patient clustering
        self.k_estimate = k_estimate # number of nearest neighbors for estimating the community
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
            Similarity_matrix_[i, idx[self.k_clustering :]] = (
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
    
    def estimate_community(self,  Patient_ids_train, Patient_ids_hat, Patient_subgroups_train, Similarity_matrix):
        """
        Estimate the patient subgroups for the new patients.
        Parameters
        ----------
        Patient_ids_train : list of str
            The patient ids in the training set.
        Patient_ids_hat : list of str
            The patient ids in the test set.
        Patient_subgroups_train : dict
            The patient subgroups in the training set. The key is the subgroup id, and the value is a list of patient ids.
        Similarity_matrix : numpy array (n_patients_train, n_patients_hat)
            The similarity matrix between the training set and the test set.
        Returns
        -------
        Patient_subgroups_hat : dict
            The estimated patient subgroups. The key is the subgroup id, and the value is a list of patient ids.
    
        """
        Labels_train = np.zeros(Similarity_matrix.shape[0], dtype=int)
        Labels_hat = np.zeros(Similarity_matrix.shape[1], dtype=int)
        for group_id in Patient_subgroups_train:
            for patient_id in Patient_subgroups_train[group_id]:
                Labels_train[Patient_ids_train.index(patient_id)] = group_id
        for new_patient in range(Similarity_matrix.shape[1]):
            idx = np.argsort(Similarity_matrix[:, new_patient])[
                ::-1
            ]
            knn_idx = idx[:self.k_estimate]
            knn_labels = Labels_train[knn_idx]
            knn_similarities = Similarity_matrix[knn_idx, new_patient]
            unique, counts = np.unique(knn_labels, return_counts=True)
            similarities_within_knn = np.zeros(len(unique))
            for j in range(len(unique)):
                similarities_within_knn[j] = np.sum(knn_similarities[knn_labels == unique[j]])
            Labels_hat[new_patient] = unique[np.argmax(similarities_within_knn)]
        Patient_subgroups_hat = {i: [] for i in range(len(Patient_subgroups_train))}
        for i in range(len(Labels_hat)):
            Patient_subgroups_hat[int(Labels_hat[i])].append(Patient_ids_hat[i])
        return Patient_subgroups_hat
        

