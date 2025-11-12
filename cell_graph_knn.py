from sklearn.neighbors import NearestNeighbors
import numpy as np

class Cell_Graph_KNN(object):
    """
    Drop-in replacement for Cell_Graph that builds a KNN cell graph:
      - same public API as cell_graph.Cell_Graph
      - returns (patient_id, adj, x) with adj as dense NxN (zeros for non-edges)
      - weights use the same Gaussian kernel: w_ij = exp(-a * d_ij^2)
    """
    def __init__(self, a=0.01, k_cell_knn=20, symmetric=True):
        self.a = a
        self.k = k_cell_knn
        self.symmetric = symmetric

    def _pos2adj_knn(self, pos: np.ndarray) -> np.ndarray:
        """
        Build a KNN adjacency (dense NxN with zeros on non-edges).
        pos: (N,2) array of (x,y)
        """
        n = pos.shape[0]
        if n <= 1:
            return np.zeros((n, n), dtype=np.float32)

        # k cannot exceed n-1
        k = int(min(self.k, max(n - 1, 1)))

        # fit KNN on coordinates
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")  # +1 includes self
        nbrs.fit(pos)
        dists, inds = nbrs.kneighbors(pos, return_distance=True)

        # Allocate dense adj and fill the KNN edges
        adj = np.zeros((n, n), dtype=np.float32)
        # skip column 0 (self), use columns 1..k for neighbors
        nb_idx = inds[:, 1:]
        nb_dist = dists[:, 1:]

        # Gaussian weight
        weights = np.exp(-self.a * (nb_dist ** 2)).astype(np.float32)

        # fill row-wise
        rows = np.repeat(np.arange(n), k)
        cols = nb_idx.ravel()
        adj[rows, cols] = weights.ravel()

        # symmetrize if requested
        if self.symmetric:
            adj = np.maximum(adj, adj.T)

        # zero diagonal
        np.fill_diagonal(adj, 0.0)
        return adj

    def one_hot_encode(self, x, dimension):
        n = len(x)
        one_hot = np.zeros((n, dimension), dtype=np.float32)
        one_hot[np.arange(n), x] = 1.0
        return one_hot

    def block_diagonal(self, arrays):
        sizes = [a.shape[1] for a in arrays]
        out = np.zeros((sum(sizes), sum(sizes)), dtype=np.float32)
        offset = 0
        for a, size in zip(arrays, sizes):
            out[offset:offset + a.shape[0], offset:offset + size] = a
            offset += size
        return out

    def merge_cell_graphs(self, cell_graphs):
        patient_id = cell_graphs[0][0]
        adjs = [cg[1] for cg in cell_graphs]
        xs   = [cg[2] for cg in cell_graphs]
        adj = self.block_diagonal(adjs)
        x   = np.concatenate(xs, axis=0)
        return (patient_id, adj, x)

    def generate(self, singleCell_data):
        """
        Return list of (patient_id, adj, x), like the original Cell_Graph.generate
        but with KNN-based adjacencies (dense with zeros for non-neighbors).
        """
        Cell_graphs = []
        Patient_ids = singleCell_data["patientID"].unique()
        num_unique_cell_types = int(singleCell_data["celltypeID"].max()) + 1

        for patient_id in Patient_ids:
            patient_data = singleCell_data[singleCell_data["patientID"] == patient_id]
            if patient_data["imageID"].nunique() > 1:
                image_ids = patient_data["imageID"].unique()
                parts = []
                for image_id in image_ids:
                    image_data = patient_data[patient_data["imageID"] == image_id]
                    pos = image_data[["coorX", "coorY"]].values.astype(np.float32)
                    adj = self._pos2adj_knn(pos)
                    x   = self.one_hot_encode(image_data["celltypeID"].to_numpy(), num_unique_cell_types)
                    parts.append((patient_id, adj, x))
                cell_graph = self.merge_cell_graphs(parts)
            else:
                pos = patient_data[["coorX", "coorY"]].values.astype(np.float32)
                adj = self._pos2adj_knn(pos)
                x   = self.one_hot_encode(patient_data["celltypeID"].to_numpy(), num_unique_cell_types)
                cell_graph = (patient_id, adj, x)

            Cell_graphs.append(cell_graph)

        return Cell_graphs
