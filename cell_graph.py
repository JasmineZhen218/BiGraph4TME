from scipy.spatial.distance import pdist, squareform
import numpy as np


class Cell_Graph:
    def __init__(self, a=0.01):
        self.a = a  # parameter for edge weight calculation

    def Pos2Adj(self, pos):
        """
        Convert the position of cells to the adjacency matrix of the cell graph.
        parameters
        ----------
        pos : numpy array Nx2
            The position of cells. Each row is the (x, y) coordinate of a cell.
        Returns
        -------
        adj : numpy array NxN
            The adjacency matrix of the cell graph.
        """
        distance = squareform(
            pdist(pos)
        )  # Euclidean distance between all pairs of points
        adj = np.exp(
            -self.a * distance * distance
        )  # edge weight = exp(- a * distance^2)
        return adj

    def block_digonal(self, arrays):
        """
        Concatenate arrays along the diagonal.
        parameters
        ----------
        arrays : list of numpy arrays
            The input arrays.
        Returns
        -------
        block_digonal : numpy array
            The concatenated array.
        Example
            -------
            A = np.array([[1, 2],
                [3, 4]])
            B = np.array([[5, 6],
                        [7, 8]])
            C = np.array([[9, 10, 0, 0],
                        [11, 12, 0, 0],
                        [0, 0, 13, 14],
                        [0, 0, 15, 16]])

            result = diagonal_array([A, B, C])
            print(result)
           [[ 1  2  0  0  0  0  0  0]
            [ 3  4  0  0  0  0  0  0]
            [ 0  0  5  6  0  0  0  0]
            [ 0  0  7  8  0  0  0  0]
            [ 0  0  0  0  9 10  0  0]
            [ 0  0  0  0 11 12  0  0]
            [ 0  0  0  0  0  0 13 14]
            [ 0  0  0  0  0  0 15 16]]
        """
        return np.block(
            [
                [
                    np.zeros(
                        (
                            array.shape[0],
                            sum(a.shape[1] for a in arrays if array is not a),
                        )
                    ),
                    array,
                    np.zeros(
                        (
                            array.shape[0],
                            sum(a.shape[1] for a in arrays if array is not a),
                        )
                    ),
                ]
                for array in arrays
            ]
        )

    def merge_cell_graphs(self, cell_graphs):
        """
        Merge cell graphs of the same patient.
        parameters
        ----------
        cell_graphs : list of tuples
            Each tuple is a cell graph, with the first element being the patient id, the second element being the adjacency matrix, and the third element being the cell type id array
        Returns
        -------
        cell_graph : tuple
            The merged cell graph, with the first element being the patient id, the second element being the concatenated adjacency matrix, and the third element being the concatenated cell type id array
        """
        patient_id = cell_graphs[0][0]
        adjs = [cell_graph[1] for cell_graph in cell_graphs]
        xs = [cell_graph[2] for cell_graph in cell_graphs]
        adj = self.block_digonal(adjs)
        x = np.concatenate(xs, axis=0)
        return (patient_id, adj, x)

    def generate(self, singleCell_data):
        """
        Generate cell graphs from single cell data.
        parameters
        ----------
        singleCell_data : pandas dataframe
            The input data, with columns specify patient id, image id, cell type id, x coordinates, y coordinates.
        Returns
        -------
        Cell_graphs : list of tuples
            Each tuple is a cell graph, with the first element being the patient id, the second element being the adjacency matrix, and the third element being the cell type id array
        """
        Cell_graphs = []
        Patient_ids = singleCell_data["patientID"].unique()
        for patient_id in Patient_ids:
            patient_data = singleCell_data[singleCell_data["patientID"] == patient_id]
            if patient_data["imageID"].nunique() > 1:
                print(
                    f"Warning: patient {patient_id} has {patient_data['imageID'].nunique()} images."
                )
                image_ids = patient_data["imageID"].unique()
                cell_graphs = []
                for image_id in image_ids:
                    image_data = patient_data[patient_data["imageID"] == image_id]
                    pos = image_data[["coorX", "coorY"]].values
                    adj = self.Pos2Adj(pos)
                    x = image_data["celltypeID"].values
                    cell_graphs.append((patient_id, adj, x))
                cell_graph = self.merge_cell_graphs(cell_graphs)

            else:
                pos = patient_data[["coorX", "coorY"]].values
                adj = self.Pos2Adj(pos)
                x = patient_data["celltypeID"].values
                cell_graph = (patient_id, adj, x)
            Cell_graphs.append(cell_graph)
        return Cell_graphs
