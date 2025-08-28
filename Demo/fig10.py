import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from utils import reverse_dict, preprocess_Danenberg
from definitions import get_node_id, get_node_color, Cell_types_displayed_Danenberg
sys.path.append("./..")
from bi_graph import BiGraph
from explainer import Explainer

# --- Setup ---
os.makedirs("Results/Fig10", exist_ok=True)
# Load and process data
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
pop_graph, subgroups = bigraph_.fit_transform(SC_d, survival_data=survival_d)
patient_ids = list(SC_d["patientID"].unique())
X_prime = bigraph_.fitted_soft_wl_subtree.X_prime
Signatures = bigraph_.fitted_soft_wl_subtree.Signatures
explainer_ = Explainer()

# Utility functions
def get_pos(G):
    return {k: (G.nodes[k]["coorX"], G.nodes[k]["coorY"]) for k in G.nodes}

def normalize_pos(pos):
    x, y = np.array(list(pos.values())).T
    x -= np.mean(x)
    y -= np.mean(y)
    return {k: (x[i], y[i]) for i, k in enumerate(pos)}

def get_subtree_leaves(root_idx, Adj, iterations=2, threshold=0.1):
    W = Adj.copy()
    for _ in range(iterations):
        W = W @ Adj
    return list(np.where(W[root_idx] > threshold)[0])

def plot_subtree(patient_id, adj, cells, root_idx, leaves, color_map, save_path):
    adj_ = np.where(adj < 0.01, 0, adj.copy())
    np.fill_diagonal(adj_, 0)
    G = nx.from_numpy_array(adj_)
    nx.set_node_attributes(G, cells["coorX"].to_dict(), "coorX")
    nx.set_node_attributes(G, cells["coorY"].to_dict(), "coorY")
    nx.set_node_attributes(G, cells["celltypeID"].to_dict(), "celltypeID")
    nx.set_node_attributes(G, cells["meta_description"].to_dict(), "meta_description")

    subG = G.subgraph(leaves)
    pos = normalize_pos(get_pos(subG))
    edge_weights = [10 * subG[u][v].get("weight", 0) for u, v in subG.edges()]
    cmap = mcolors.LinearSegmentedColormap.from_list("greys", plt.get_cmap("Greys")(np.linspace(0.3, 1.0, 256)))

    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw(
        subG,
        pos,
        node_size=600,
        node_color=[color_map[d] for d in nx.get_node_attributes(subG, "meta_description").values()],
        labels=nx.get_node_attributes(subG, "celltypeID"),
        font_color="k",
        edgecolors="k",
        edge_color=edge_weights,
        edge_cmap=cmap,
        width=2,
        ax=ax,
    )
    ax.add_patch(plt.Circle(pos[root_idx], 5, color="k", fill=False, linestyle="--"))
    fig.savefig(f"{save_path}.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Plot characteristic subtrees
for subgroup in subgroups:
    sid = subgroup["subgroup_id"]
    for pattern_id in subgroup["characteristic_patterns"]:
        reps = explainer_.find_representative_subtrees(X_prime, Signatures, pattern_id, n=10)
        for idx, (pid, root_idx) in enumerate(reps):
            cells = SC_d[SC_d.patientID == pid].reset_index(drop=True)
            adj = X_prime[patient_ids.index(pid)][1]
            leaves = get_subtree_leaves(root_idx, adj)
            color_map = get_node_color("Danenberg", "CellType")
            save_path = f"Results/Fig10/fig10_{sid}_{pattern_id}_{idx}"
            plot_subtree(pid, adj, cells, root_idx, leaves, color_map, save_path)

# Legend
fig, ax = plt.subplots(figsize=(10, 0.4), tight_layout=True)
labels = Cell_types_displayed_Danenberg
colors = [get_node_color("Danenberg", "CellType")[reverse_dict(get_node_id("Danenberg", "CellType"))[i]] for i in range(len(labels))]
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8) for c in colors]
ax.legend(handles, labels, loc=3, framealpha=0.5, frameon=True, ncols=3, bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", fontsize=10)
ax.axis("off")
fig.savefig("Results/Fig10/fig10_legend.svg", dpi=300, bbox_inches="tight")
