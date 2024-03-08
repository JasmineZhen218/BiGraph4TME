import pandas as pd
from soft_wl_subtree import Soft_WL_Subtree
from cell_graph import Cell_Graph
from population_graph import Population_Graph
from explainer import Explainer


class BiGraph:
    def __init__(self, a = 0.01, n_iter=0, k_subtree_clustering=100, k_patient_clustering=30):
        self.a = a  # parameter for edge weight calculation in cell graph  $w_{ij} = \exp(-a \cdot d_{ij}^2)$
        self.n_iter = n_iter  # number of iterations
        self.k_subtree_clustering = (
            k_subtree_clustering  # decides the coarseness of subtree clustering
        )
        self.k_patient_clustering = (
            k_patient_clustering  # decides the coarseness of patient clustering
        )

    def fit_transform(
        self,
        singleCell_data,
        patientID_colname="patientID",
        imageID_colname="imageID",
        celltypeID_colname="celltypeID",
        coorX_colname="coorX",
        coorY_colname="coorY",
    ):
        """
        Parameters
        ----------
        singleCell_data : pandas dataframe
            The input data, with columns specify patient id, image id, cell type id, x coordinates, y coordinates.
        patientID_colname : str, optional
            The column name of patient id. The default is "patientID".
        imageID_colname : str, optional
            The column name of image id. The default is "imageID".
        celltypeID_colname : str, optional
            The column name of cell type id. The default is "celltypeID".
        coorX_colname : str, optional
            The column name of x coordinates. The default is "coorX".
        coorY_colname : str, optional
            The column name of y coordinates. The default is "coorY".

        Returns
        -------
        Population_graph : networkx graph
            The population graph.
        Patient_subgroups : dict
            The patient subgroups. The key is the subgroup id, and the value is a list of patient ids.
        Characteristic_patterns : dict
            The characteristic patterns for each patient subgroup. The key is the subgroup id, and the value is a list of TME pattern id.
        """
        singleCell_data = pd.rename(
            columns={
                patientID_colname: "patientID",
                imageID_colname: "imageID",
                celltypeID_colname: "celltypeID",
                coorX_colname: "coorX",
                coorY_colname: "coorY",
            }
        )
        print("Basic data preprocessing done. An overview of the data is as follows: ")
        print(
            "\t {} unique patients.".format(len(singleCell_data["patientID"].unique()))
        )
        print(
            "\t {} unique cell types.".format(
                len(singleCell_data["celltypeID"].unique())
            )
        )
        print(
            "\t {} total cells, {} cells per patient.".format(
                len(singleCell_data),
                len(singleCell_data) / len(singleCell_data["patientID"].unique()),
            )
        )
        cell_graph_ = Cell_Graph(a = self.a)
        Cell_graphs = cell_graph_.generate(singleCell_data)
        soft_wl_subtree_ = Soft_WL_Subtree(n_iter=self.n_iter, k=self.k_subtree_clustering)
        Similarity_matrix = soft_wl_subtree_.fit_transform(Cell_graphs)
        population_graph_ = Population_Graph()
        Population_graph = population_graph_.generate(Similarity_matrix)
        Patient_subgroups = population_graph_.community_detection(Population_graph)
        explainer_ = Explainer()
        Characteristic_patterns = explainer_.fit_transform()

        self.soft_wl_subtree_ = soft_wl_subtree_
        self.population_graph_ = population_graph_
        return Population_graph, Patient_subgroups, Characteristic_patterns

    def transform(
        self,
        singleCell_data,
        patientID_colname="patientID",
        imageID_colname="imageID",
        celltypeID_colname="celltypeID",
        coorX_colname="coorX",
        coorY_colname="coorY",
    ):
        """
        Parameters
        ----------
        singleCell_data : pandas dataframe
            The input data, with columns specify patient id, image id, cell type id, x coordinates, y coordinates.
        patientID_colname : str, optional

            The column name of patient id. The default is "patientID".
        imageID_colname : str, optional
            The column name of image id. The default is "imageID".
        celltypeID_colname : str, optional
            The column name of cell type id. The default is "celltypeID".
        coorX_colname : str, optional
            The column name of x coordinates. The default is "coorX".
        coorY_colname : str, optional
            The column name of y coordinates. The default is "coorY".

        Returns
        -------
        Patient_subgroups_hat : dict
            The estimated patient subgroups. The key is the subgroup id, and the value is a list of patient ids.
        """

        singleCell_data = pd.rename(
            columns={
                patientID_colname: "patientID",
                imageID_colname: "imageID",
                celltypeID_colname: "celltypeID",
                coorX_colname: "coorX",
                coorY_colname: "coorY",
            }
        )
        print("Basic data preprocessing done. An overview of the data is as follows: ")
        print(
            "\t {} unique patients.".format(len(singleCell_data["patientID"].unique()))
        )
        print(
            "\t {} unique cell types.".format(
                len(singleCell_data["celltypeID"].unique())
            )
        )
        print(
            "\t {} total cells, {} cells per patient.".format(
                len(singleCell_data),
                len(singleCell_data) / len(singleCell_data["patientID"].unique()),
            )
        )
        pass
        cell_graph_ = Cell_Graph()
        Cell_graphs = cell_graph_.generate(singleCell_data)
        Similarity_matrix = self.soft_wl_subtree.transform(Cell_graphs)
        Patient_subgroups_hat = self.population_graph_.estimate_community(
            Similarity_matrix
        )
        return Patient_subgroups_hat
