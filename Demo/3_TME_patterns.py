import pandas as pd
import random
from definitions import get_node_color, get_node_id
import sys
import numpy as np
import os
import argparse
from utils import PROJECT_ROOT
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cell_graph import Cell_Graph
from soft_wl_subtree import Soft_WL_Subtree

"""
set export OMP_NUM_THREADS=1 to avoid memory error
"""
os.system("export OMP_NUM_THREADS=1")

# read single cell data and clinical data
cells = pd.read_csv(os.path.join(PROJECT_ROOT, "Datasets/Danenberg_et_al/cells.csv"))
clinical = pd.read_csv(
    os.path.join(PROJECT_ROOT, "Datasets/Danenberg_et_al/clinical.csv")
)
print("Initially,")
print(
    "{} patients ({} images) with cell data, {} patients with clinical data, ".format(
        len(cells["metabric_id"].unique()),
        len(cells["ImageNumber"].unique()),
        len(clinical["metabric_id"].unique()),
    )
)
# remove images without invasive tumor
print("\nRemove images without invasive tumor,")
cells = cells.loc[cells.isTumour == 1]
print(
    "{} patients ({} images) with cell data, {} patients with clinical data, ".format(
        len(cells["metabric_id"].unique()),
        len(cells["ImageNumber"].unique()),
        len(clinical["metabric_id"].unique()),
    )
)
# remove patients with no clinical data
print("\nRemove patients with no clinical data,")
cells = cells.loc[cells["metabric_id"].isin(clinical["metabric_id"])]
print(
    "{} patients ({} images) with cell data and clinical data, ".format(
        len(cells["metabric_id"].unique()),
        len(cells["ImageNumber"].unique()),
    )
)
# remove images with less than 500 cells
print("\nRemove images with less than 500 cells")
cells_per_image = cells.groupby("ImageNumber").size()
cells = cells.loc[
    cells["ImageNumber"].isin(cells_per_image[cells_per_image > 500].index)
]
clinical = clinical.loc[clinical["metabric_id"].isin(cells["metabric_id"].unique())]
print(
    "{} patients ({} images) with more than 500 cells and clinical data, ".format(
        len(cells["metabric_id"].unique()),
        len(cells["ImageNumber"].unique()),
    )
)

random.seed(0)
Subset_id = [1] * (len(clinical) - 200) + [2] * 200
random.shuffle(Subset_id)
clinical["Subset_id"] = Subset_id
cells_discovery = cells.loc[
    cells["metabric_id"].isin(clinical.loc[clinical["Subset_id"] == 1, "metabric_id"])
]
cells_validation = cells.loc[
    cells["metabric_id"].isin(clinical.loc[clinical["Subset_id"] == 2, "metabric_id"])
]
print("\nAfter splitting into discovery and validation sets,")
print(
    "{} patients ({} images) with more than 500 cells and clinical data in the discovery set, ".format(
        len(cells_discovery["metabric_id"].unique()),
        len(cells_discovery["ImageNumber"].unique()),
    )
)

# Assign cell type Id based on meta description column
cells["cellTypeID"] = cells["meta_description"].map(
    get_node_id("Danenberg", "CellType")
)
# standardize column names
patientID_colname = "metabric_id"
imageID_colname = "ImageNumber"
celltypeID_colname = "cellTypeID"
coorX_colname = "Location_Center_X"
coorY_colname = "Location_Center_Y"
cells = cells.rename(
    columns={
        patientID_colname: "patientID",
        imageID_colname: "imageID",
        celltypeID_colname: "celltypeID",
        coorX_colname: "coorX",
        coorY_colname: "coorY",
    }
)

cell_graph_ = Cell_Graph(a=0.01)
Cell_graphs = cell_graph_.generate(cells)
print("There are {} patients/cell graphs".format(len(Cell_graphs)))


cell_graph = Cell_graphs[0]
print(
    "The first cell graph is a tuple with 3 elements: (patient_id, graph, cell_types)"
)
print("\tThe first element is the patient id: {}".format(cell_graph[0]))
print(
    "\tThe second element is the adjacnecy matrix, with the shape of {}".format(
        cell_graph[1].shape
    )
)
print(
    "\tThe third element is the cell types, with the shape of {}".format(
        cell_graph[2].shape
    )
)
print(
    "There are {} cells with {} unique cell types".format(
        cell_graph[1].shape[0], np.unique(np.where(cell_graph[2] == 1)[1]).shape[0]
    )
)
# for debug
Cell_graphs = Cell_graphs[:10]
soft_wl_subtree_ = Soft_WL_Subtree(n_iter=2, k=100, n_jobs=1)
Similarity_matrix = soft_wl_subtree_.fit_transform(Cell_graphs)
print("The similarity matrix has a shape of {}.",format(Similarity_matrix.shape))
Signatures = soft_wl_subtree_.Signatures
print("There are {} discovered patterns".format(len(Signatures)))
Cell_graphs_prime = soft_wl_subtree_.X_prime
cell_graph_prime = Cell_graphs_prime[0]
print(
    "The first Cell_graphs_prime element (and all others) is a tuple: (patient_id, adj, patterns)"
)
print("\tThe first element is the patient id: {}".format(cell_graph_prime[0]))
print(
    "\tThe second element is the adjacnecy matrix, with the shape of {}".format(
        cell_graph_prime[1].shape
    )
)
print(
    "\tThe third element is the patterns, with the shape of {}".format(
        cell_graph_prime[2].shape
    )
)
print(
    "There are {} cells with {} unique patterns".format(
        cell_graph_prime[1].shape[0], np.unique(cell_graph_prime[2]).shape[0]
    )
)

# save the fitted soft_wl_subtree_ model
import pickle

with open("fitted_soft_wl_subtree.pkl", "wb") as f:
    pickle.dump(soft_wl_subtree_, f)
print("The fitted soft_wl_subtree_ model is saved as fitted_soft_wl_subtree.pkl")
