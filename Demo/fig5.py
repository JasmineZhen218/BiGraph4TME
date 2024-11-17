import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from utils import  preprocess_Danenberg
from definitions import get_node_color
sys.path.append("./..")
from bi_graph import BiGraph

SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)

Similarity_matrix = bigraph_.fitted_soft_wl_subtree.Similarity_matrix
Cellular_graphs = bigraph_.fitted_soft_wl_subtree.X
patient_ids_d = [cellular_graph[0] for cellular_graph in Cellular_graphs]
print("The shape of the similarity matrix is", Similarity_matrix.shape)
print("There are {} patients".format(len(patient_ids_d)))
def get_pos(G):
    x = nx.get_node_attributes(G, "coorX")
    y = nx.get_node_attributes(G, "coorY")
    pos = {}
    for key, _ in x.items():
        pos[key] = (x[key], y[key])
    return pos

template_patient_id = "MB-0882"
template_image_id = 50
adj = Cellular_graphs[patient_ids_d.index(template_patient_id)][
    1
]  # The adjacency matrix of the template patient
cells_template = SC_d[SC_d["imageID"] == template_image_id].reset_index(drop=True)
assert len(cells_template) == adj.shape[0]
print("The number of cells in the template patient is", len(cells_template))
# For visualization propose, we set the diagonal of the adjacency matrix to be 0
np.fill_diagonal(adj, 0)  # Remove self loops
# For visualization purpose, we only show edges with a weight higher than 0.01
adj[adj < 0.01] = 0
G = nx.from_numpy_array(adj)
nx.set_node_attributes(G, cells_template["coorX"], "coorX")
nx.set_node_attributes(G, cells_template["coorY"], "coorY")
nx.set_node_attributes(G, cells_template["meta_description"], "meta_description")
f, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    tight_layout=True,
)
cell_color_dict = get_node_color("Danenberg", "CellType")
nx.draw_networkx_nodes(
    G,
    get_pos(G),
    node_size=18,
    node_color=[
        cell_color_dict[cell_type]
        for cell_type in list(nx.get_node_attributes(G, "meta_description").values())
    ],
    # edgecolors="black",
    linewidths=0.5,
    ax=ax,
)
nx.draw_networkx_edges(G, get_pos(G), ax=ax, alpha=0.3, width=1)
ax.set(
    title="Template patient",
    xlim=(
        (max(cells_template["coorX"]) - min(cells_template["coorX"])) / 2 - 350,
        (max(cells_template["coorX"]) - min(cells_template["coorX"])) / 2 + 350,
    ),
    ylim=(
        (max(cells_template["coorY"]) - min(cells_template["coorY"])) / 2 - 350,
        (max(cells_template["coorY"]) - min(cells_template["coorY"])) / 2 + 350,
    ),
)
ax.set_axis_off()
f.savefig("Results/fig5_a.jpg", dpi=300, bbox_inches="tight")

template_patient_index = patient_ids_d.index(
    template_patient_id
)  # The index of the template patient in the similarity matrix
Similarities_sorted = np.sort(Similarity_matrix[template_patient_index, :])[
    ::-1
]  # Sort the similarities in descending order
Representative_patient_ids = ["MB-0262", "MB-0356", "MB-0290"]
f, ax = plt.subplots(figsize=(8, 1.5))
ax.plot(Similarities_sorted[:100], color="k")
# TEMPLATE PATIENT
ax.scatter(0, Similarities_sorted[0], color="red", s=8)
# REPRESENTATIVE PATIENTS
Representative_patient_indices = (
    patient_ids_d.index(Representative_patient_ids[0]),
    patient_ids_d.index(Representative_patient_ids[1]),
    patient_ids_d.index(Representative_patient_ids[2]),
)
Patient_indices_sorted = np.argsort(Similarity_matrix[template_patient_index, :])[::-1]
Representative_patient_rankings = [
    list(Patient_indices_sorted).index(i) for i in Representative_patient_indices
]
for indice in Representative_patient_rankings:
    ax.scatter(indice, Similarities_sorted[indice], color="red", s=8)
    print(
        "Patient ID:",
        patient_ids_d[indice],
        "Similarity:",
        Similarities_sorted[indice],
        "Rank:",
        indice,
    )
ax.set_xlabel("Patient Rank", fontsize=12)
ax.set_ylabel("Similarity", fontsize=12)
f.savefig("Results/fig5_c.jpg", dpi=300, bbox_inches="tight")

patient_id = "MB-0262"
image_id = SC_d[SC_d["patientID"] == patient_id]["imageID"].values[0]
print("The image ID of the patient", patient_id, "is", image_id)
adj = Cellular_graphs[patient_ids_d.index(patient_id)][
    1
]  # The adjacency matrix of the template patient
cells_ = SC_d[SC_d["imageID"] == image_id].reset_index(drop=True)
assert len(cells_) == adj.shape[0]
print("The number of cells in the template patient is", len(cells_))
np.fill_diagonal(adj, 0)  # Remove self loops
adj[adj < 0.01] = 0
G = nx.from_numpy_array(adj)
nx.set_node_attributes(G, cells_["coorX"], "coorX")
nx.set_node_attributes(G, cells_["coorY"], "coorY")
nx.set_node_attributes(G, cells_["meta_description"], "meta_description")
f, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    tight_layout=True,
)
cell_color_dict = get_node_color("Danenberg", "CellType")
nx.draw_networkx_nodes(
    G,
    get_pos(G),
    node_size=18,
    node_color=[
        cell_color_dict[cell_type]
        for cell_type in list(nx.get_node_attributes(G, "meta_description").values())
    ],
    # edgecolors="black",
    linewidths=0.5,
    ax=ax,
)
nx.draw_networkx_edges(G, get_pos(G), ax=ax, alpha=0.3, width=1)
ax.set(
    title="Similarity = {:.2f}".format(
        Similarity_matrix[template_patient_index, patient_ids_d.index(patient_id)]
    ),
    xlim=(
        (max(cells_["coorX"]) - min(cells_["coorX"])) / 2 - 350,
        (max(cells_["coorX"]) - min(cells_["coorX"])) / 2 + 350,
    ),
    ylim=(
        (max(cells_["coorY"]) - min(cells_["coorY"])) / 2 - 350,
        (max(cells_["coorY"]) - min(cells_["coorY"])) / 2 + 350,
    ),
)
ax.set_axis_off()
f.savefig("Results/fig5_d.jpg", dpi=300, bbox_inches="tight")

patient_id = "MB-0356"
image_id = SC_d[SC_d["patientID"] == patient_id]["imageID"].values[0]
print("The image ID of the patient", patient_id, "is", image_id)
adj = Cellular_graphs[patient_ids_d.index(patient_id)][
    1
]  # The adjacency matrix of the template patient
cells_ = SC_d[SC_d["imageID"] == image_id].reset_index(drop=True)
assert len(cells_) == adj.shape[0]
print("The number of cells in the template patient is", len(cells_))
np.fill_diagonal(adj, 0)  # Remove self loops
adj[adj < 0.01] = 0
G = nx.from_numpy_array(adj)
nx.set_node_attributes(G, cells_["coorX"], "coorX")
nx.set_node_attributes(G, cells_["coorY"], "coorY")
nx.set_node_attributes(G, cells_["meta_description"], "meta_description")
f, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    tight_layout=True,
)
cell_color_dict = get_node_color("Danenberg", "CellType")
nx.draw_networkx_nodes(
    G,
    get_pos(G),
    node_size=18,
    node_color=[
        cell_color_dict[cell_type]
        for cell_type in list(nx.get_node_attributes(G, "meta_description").values())
    ],
    # edgecolors="black",
    linewidths=0.5,
    ax=ax,
)
nx.draw_networkx_edges(G, get_pos(G), ax=ax, alpha=0.3, width=1)
ax.set(
    title="Similarity = {:.2f}".format(
        Similarity_matrix[template_patient_index, patient_ids_d.index(patient_id)]
    ),
    xlim=(
        (max(cells_["coorX"]) - min(cells_["coorX"])) / 2 - 350,
        (max(cells_["coorX"]) - min(cells_["coorX"])) / 2 + 350,
    ),
    ylim=(
        (max(cells_["coorY"]) - min(cells_["coorY"])) / 2 - 350,
        (max(cells_["coorY"]) - min(cells_["coorY"])) / 2 + 350,
    ),
)
ax.set_axis_off()
f.savefig("Results/fig5_e.jpg", dpi=300, bbox_inches="tight")

patient_id = "MB-0290"
image_id = SC_d[SC_d["patientID"] == patient_id]["imageID"].values[0]
print("The image ID of the patient", patient_id, "is", image_id)
adj = Cellular_graphs[patient_ids_d.index(patient_id)][
    1
]  # The adjacency matrix of the template patient
cells_ = SC_d[SC_d["imageID"] == image_id].reset_index(drop=True)
assert len(cells_) == adj.shape[0]
print("The number of cells in the template patient is", len(cells_))
np.fill_diagonal(adj, 0)  # Remove self loops
adj[adj < 0.01] = 0
G = nx.from_numpy_array(adj)
nx.set_node_attributes(G, cells_["coorX"], "coorX")
nx.set_node_attributes(G, cells_["coorY"], "coorY")
nx.set_node_attributes(G, cells_["meta_description"], "meta_description")
f, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    tight_layout=True,
)
cell_color_dict = get_node_color("Danenberg", "CellType")
nx.draw_networkx_nodes(
    G,
    get_pos(G),
    node_size=18,
    node_color=[
        cell_color_dict[cell_type]
        for cell_type in list(nx.get_node_attributes(G, "meta_description").values())
    ],
    # edgecolors="black",
    linewidths=0.5,
    ax=ax,
)
nx.draw_networkx_edges(G, get_pos(G), ax=ax, alpha=0.3, width=1)
ax.set(
    title="Similarity = {:.2f}".format(
        Similarity_matrix[template_patient_index, patient_ids_d.index(patient_id)]
    ),
    xlim=(
        (max(cells_["coorX"]) - min(cells_["coorX"])) / 2 - 350,
        (max(cells_["coorX"]) - min(cells_["coorX"])) / 2 + 350,
    ),
    ylim=(
        (max(cells_["coorY"]) - min(cells_["coorY"])) / 2 - 350,
        (max(cells_["coorY"]) - min(cells_["coorY"])) / 2 + 350,
    ),
)
ax.set_axis_off()
f.savefig("Results/fig5_f.jpg", dpi=300, bbox_inches="tight")