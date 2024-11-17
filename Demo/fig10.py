import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from utils import reverse_dict, preprocess_Danenberg
from definitions import (
    get_node_id,
    get_node_color,
)
sys.path.append("./..")
from bi_graph import BiGraph
from explainer import Explainer

SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)
patient_ids_discovery = list(SC_d["patientID"].unique())

explainer_ = Explainer()
X_prime = bigraph_.fitted_soft_wl_subtree.X_prime
Signatures = bigraph_.fitted_soft_wl_subtree.Signatures

def get_pos(G):
    x = nx.get_node_attributes(G, "coorX")
    y = nx.get_node_attributes(G, "coorY")
    pos = {}
    for key, _ in x.items():
        pos[key] = (x[key], y[key])
    return pos


def normalize_pos(pos):
    x = np.array([pos[key][0] for key in pos.keys()])
    y = np.array([pos[key][1] for key in pos.keys()])
    x = x - np.mean(x)
    y = y - np.mean(y)
    pos_normalized = {}
    for i, key in enumerate(pos.keys()):
        pos_normalized[key] = (x[i], y[i])
    return pos_normalized


def decide_subtree_boundary(root_idx, Adj, iteration, boundary_weight_threshold=0.1):
    W = Adj.copy()
    for i in range(iteration):
        W = np.matmul(W, Adj)
    leaf_indices = list(np.where(W[root_idx, :] > boundary_weight_threshold)[0])
    return leaf_indices


for i in range(len(patient_subgroups_discovery)):
    subgroup_id = patient_subgroups_discovery[i]["subgroup_id"]
    characteristic_patterns = patient_subgroups_discovery[i]["characteristic_patterns"]
    for pattern_index in range(len(characteristic_patterns)):
        pattern_id = characteristic_patterns[pattern_index]
        representative_subtrees = explainer_.find_representative_subtrees(
            X_prime, Signatures, pattern_id, n=1
        )
        for patient_id, subtree_root_local_idx in representative_subtrees:
            adj = X_prime[patient_ids_discovery.index(patient_id)][1]
            subtree_leaves_local_idx = decide_subtree_boundary(
                subtree_root_local_idx, adj, 2, boundary_weight_threshold=0.1
            )
            cells_ = SC_d.loc[SC_d.patientID == patient_id].reset_index(drop=True)
            adj_ = adj.copy()
            adj_[adj_ < 0.01] = 0
            np.fill_diagonal(adj_, 0)
            cellular_graph = nx.from_numpy_array(adj_)
            nx.set_node_attributes(cellular_graph, cells_["coorX"], "coorX")
            nx.set_node_attributes(cellular_graph, cells_["coorY"], "coorY")
            nx.set_node_attributes(cellular_graph, cells_["celltypeID"], "celltypeID")
            nx.set_node_attributes(
                cellular_graph, cells_["meta_description"], "meta_description"
            )
            subtree_graph = nx.subgraph(cellular_graph, subtree_leaves_local_idx)
            edge_list = list(subtree_graph.edges())
            edge_alpha = [
                (
                    10 * subtree_graph[u][v]["weight"]
                    if subtree_graph[u][v]["weight"] > 0.01
                    else 0
                )
                for u, v in edge_list
            ]
            edge_alpha = np.array(edge_alpha)
            cmap_greys = plt.get_cmap("Greys")
            start_point = 0.3
            end_point = 1.0
            new_cmap_colors = cmap_greys(np.linspace(start_point, end_point, 256))
            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                "Greys_RightHalf", new_cmap_colors
            )
            f, ax = plt.subplots(figsize=(5, 5))
            circle = plt.Circle(
                (
                    normalize_pos(get_pos(subtree_graph))[subtree_root_local_idx][0],
                    normalize_pos(get_pos(subtree_graph))[subtree_root_local_idx][1],
                ),
                5,
                color="k",
                fill=False,
                linestyle="--",
            )
            nx.draw_networkx(
                subtree_graph,
                normalize_pos(get_pos(subtree_graph)),
                node_size=600,
                node_color=[
                    get_node_color("Danenberg", "CellType")[cell_type]
                    for cell_type in list(
                        nx.get_node_attributes(
                            subtree_graph, "meta_description"
                        ).values()
                    )
                ],
                labels=nx.get_node_attributes(subtree_graph, "celltypeID"),
                font_color="k",
                with_labels=True,
                edgecolors="k",
                ax=ax,
                edge_color=edge_alpha,
                edge_cmap=new_cmap,
                width=2,
            )
            ax.add_patch(circle)
            ax.set_title("Characteristic in {}".format(subgroup_id), fontsize=12)
            f.savefig(
                "Results/fig10_{}_{}.jpg".format(subgroup_id, pattern_id),
                dpi=300,
                bbox_inches="tight",
            )
fig, ax = plt.subplots(figsize=(10, 0.4), tight_layout=True)
func = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none", markersize=8)[0]
cell_types = [
    str(i) + ":" + reverse_dict(get_node_id("Danenberg", "CellType"))[i]
    for i in range(len(get_node_id("Danenberg", "CellType")))
]
handles = [
    func("o", get_node_color("Danenberg", "CellType")[i.split(":")[1]])
    for i in cell_types
]
ax.legend(
    handles,
    cell_types,
    loc=3,
    framealpha=0.5,
    frameon=1,
    ncols=3,
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    mode="expand",
    borderaxespad=0.0,
    fontsize=10,
)

ax.axis("off")
plt.show()
fig.savefig("Results/fig10_legend.jpg", dpi=300, bbox_inches="tight")
