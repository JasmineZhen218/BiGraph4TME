import numpy as np
import networkx as nx

class Population_Graph_KNN:
    def __init__(
        self,
        k_clustering=20,
        k_estimate=3,
        resolution=1.0,
        size_smallest_cluster=10,
        seed=1,
    ):
        # [ORIGINAL]
        self.k_clustering = k_clustering
        self.k_estimate = k_estimate
        self.resolution = resolution
        self.seed = seed
        self.size_smallest_cluster = size_smallest_cluster

    def generate(self, Similarity_matrix, Patient_ids):
        """
        Generate the population graph using a KNN pruning (as in original) but
        with a vectorized IoU/Jaccard computation to reduce memory/time.
        """
        # [ORIGINAL]
        Similarity_matrix_ = Similarity_matrix.copy()
        np.fill_diagonal(Similarity_matrix_, 0)

        # [ORIGINAL] KNN pruning (keep top-k per row)
        for i in range(Similarity_matrix_.shape[0]):
            Similarity_matrix_[i, np.argsort(Similarity_matrix_[i, :])[: -self.k_clustering]] = 0

        # [ORIGINAL] symmetrize
        adj_1 = np.maximum(Similarity_matrix_, Similarity_matrix_.T)

        # === [NEW] Vectorized IoU/Jaccard on boolean neighbor matrix (no O(n^3) loops) ===
        # Build boolean neighbor mask
        B = (adj_1 > 0).astype(np.uint8)               # (n x n), each row <= k_clustering ones
        # Intersection counts for every (i,j)
        inter = (B @ B.T).astype(np.float64)           # (n x n)
        # Row degrees
        deg = B.sum(axis=1, dtype=np.float64)          # (n,)
        # Union = deg[i] + deg[j] - inter[i,j]
        union = deg[:, None] + deg[None, :] - inter
        # Jaccard/IoU with safe divide
        with np.errstate(divide='ignore', invalid='ignore'):
            adj_2 = np.divide(inter, union, out=np.zeros_like(inter), where=(union > 0))
        np.fill_diagonal(adj_2, 0)

        # [ORIGINAL] build graph and attach patient IDs
        G_population = nx.from_numpy_array(adj_2)
        Patient_ids_dict = {i: Patient_ids[i] for i in range(len(Patient_ids))}
        nx.set_node_attributes(G_population, Patient_ids_dict, "patientID")
        return G_population

    def community_detection(self, G_population):
        # [ORIGINAL] but pass configured seed
        Communities = nx.community.louvain_communities(
            G_population, weight="weight", resolution=self.resolution, seed=self.seed
        )
        # [ORIGINAL] size filter + sort
        Communities = [list(c) for c in Communities if len(c) > self.size_smallest_cluster]
        Communities = sorted(Communities, key=lambda x: len(x), reverse=True)

        Patient_subgroups = []
        for c in Communities:
            Patient_subgroups.append(
                {"patient_ids": [G_population.nodes[n]["patientID"] for n in c]}
            )
        return Patient_subgroups

    def estimate_community(
        self,
        Patient_ids_train,
        Patient_subgroups_train,
        Similarity_matrix,
    ):
        # [ORIGINAL] identical logic (summed-similarity k-NN voting)
        Labels_train = np.zeros(len(Patient_ids_train), dtype=object)
        for subgroup in Patient_subgroups_train:
            subgroup_id = subgroup['subgroup_id']
            for patient in subgroup["patient_ids"]:
                Labels_train[Patient_ids_train.index(patient)] = subgroup_id

        assert len(Patient_ids_train) == Similarity_matrix.shape[0]
        Labels_hat = np.zeros(Similarity_matrix.shape[1], dtype=object)

        for new_patient in range(Similarity_matrix.shape[1]):
            idx = np.argsort(Similarity_matrix[:, new_patient])[::-1]
            knn_idx = idx[: self.k_estimate]
            knn_labels = Labels_train[knn_idx]
            knn_similarities = Similarity_matrix[:, new_patient][knn_idx]
            unique = np.unique(knn_labels)
            similarities_within_knn = np.zeros(len(unique))
            for j, u in enumerate(unique):
                similarities_within_knn[j] = np.sum(knn_similarities[knn_labels == u])
            Labels_hat[new_patient] = unique[np.argmax(similarities_within_knn)]

        return Labels_hat
