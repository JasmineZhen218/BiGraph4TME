import pandas as pd
from soft_wl_subtree import Soft_WL_Subtree
from cell_graph import Cell_Graph
from population_graph import Population_Graph
from explainer import Explainer
import os
import pickle

class BiGraph(object):
    def __init__(
        self,
        a=0.01,
        n_iter=0,
        k_subtree_clustering=100,
        k_patient_clustering=30,
        resolution=1.0,
        size_smallest_cluster=10,
        threshold_hodges_lehmann=0.5,
        seed=1,
    ):
        self.a = a  # parameter for edge weight calculation in cell graph  $w_{ij} = \exp(-a \cdot d_{ij}^2)$
        self.n_iter = n_iter  # number of iterations
        self.k_subtree_clustering = (
            k_subtree_clustering  # decides the coarseness of subtree clustering
        )
        self.k_patient_clustering = (
            k_patient_clustering  # decides the coarseness of patient clustering
        )
        self.resolution = resolution  # resolution parameter for community detection
        self.size_smallest_cluster = size_smallest_cluster  # the size of the smallest patient subgroups (a subgroup smaller than that will be considered isolated patients)
        self.threshold_hodges_lehmann = threshold_hodges_lehmann
        self.seed = seed  # random seed in population graph community detection

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
        singleCell_data = singleCell_data.rename(
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
        print("Start generating cell graphs.")
        cell_graph_ = Cell_Graph(a=self.a) # initialize Cell_Graph class with parameter a
        Cell_graphs = cell_graph_.generate(singleCell_data) # generate cell graphs
        print("Cell graphs generated.")
        Patient_ids = [cell_graph[0] for cell_graph in Cell_graphs] # get patient ids
        print("Start measuring similarity between cell graphs using Soft-WL-Subtree-kernel (this is the most time-consuming step).")
        if os.path.exists("fitted_soft_wl_subtree.pkl"):
            print("Fortunately, the soft wl subtree kernel has been fitted before. We will load it directly.")
            print("If you want to re-calculate the similarity matrix, please delete the file 'fitted_soft_wl_subtree.pkl'.")
            with open("fitted_soft_wl_subtree.pkl", "rb") as f:
                soft_wl_subtree_ = pickle.load(f)
            Similarity_matrix = soft_wl_subtree_.Similarity_matrix
        else:
            soft_wl_subtree_ = Soft_WL_Subtree(
                n_iter=self.n_iter, k=self.k_subtree_clustering
            ) # initialize Soft_WL_Subtree class with parameters n_iter and k
            Similarity_matrix = soft_wl_subtree_.fit_transform(Cell_graphs) # calculate similarity matrix using Soft_WL_Subtree
        print("Similarity matrix calculated.")
        print("Start generating population graph.")
        population_graph_ = Population_Graph(
            k=self.k_patient_clustering,
            resolution=self.resolution,
            size_smallest_cluster=self.size_smallest_cluster,
            seed=self.seed,
        ) # initialize Population_Graph class with parameters k, resolution, size_smallest_cluster, and seed
        Population_graph = population_graph_.generate(Similarity_matrix, Patient_ids) # generate population graph
        Patient_subgroups = population_graph_.community_detection(Population_graph) # detect patient subgroups
        print("Population graph generated.")
        print("Start finding characteristic patterns for each patient subgroup.")
        explainer_ = Explainer(threshold_hodges_lehmann = self.threshold_hodges_lehmann) # initialize Explainer class
        Characteristic_patterns = explainer_.find_characteristic_patterns(Patient_ids, Patient_subgroups, soft_wl_subtree_.Histograms)
        self.Patient_subgroups = Patient_subgroups
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
            Similarity_matrix, self.Patient_subgroups
        )
        return Patient_subgroups_hat
