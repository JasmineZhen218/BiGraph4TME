import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from utils import preprocess_Danenberg
from definitions import get_node_color
sys.path.extend(["./..", "./../.."])
from bi_graph import BiGraph

# --- Setup ---
os.makedirs("Results/Fig5", exist_ok=True)

# --- Data Load ---
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)

# --- Model Fit ---
bigraph = BiGraph(k_patient_clustering=30)
population_graph, patient_subgroups = bigraph.fit_transform(SC_d, survival_data=survival_d)

Similarity_matrix = bigraph.fitted_soft_wl_subtree.Similarity_matrix
Cellular_graphs = bigraph.fitted_soft_wl_subtree.X
patient_ids = [g[0] for g in Cellular_graphs]

print("Similarity matrix shape:", Similarity_matrix.shape)
print("Number of patients:", len(patient_ids))

# --- Utilities ---
def get_pos(G):
    x, y = nx.get_node_attributes(G, "coorX"), nx.get_node_attributes(G, "coorY")
    return {k: (x[k], y[k]) for k in x}

def draw_graph(adj, cells, filename):
    np.fill_diagonal(adj, 0)
    adj[adj < 0.01] = 0
    G = nx.from_numpy_array(adj)
    nx.set_node_attributes(G, cells["coorX"], "coorX")
    nx.set_node_attributes(G, cells["coorY"], "coorY")
    nx.set_node_attributes(G, cells["meta_description"], "meta_description")

    f, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    pos = get_pos(G)
    colors = get_node_color("Danenberg", "CellType")
    node_colors = [colors[desc] for desc in nx.get_node_attributes(G, "meta_description").values()]

    nx.draw_networkx_nodes(G, pos, node_size=18, node_color=node_colors, linewidths=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1)

    x_mid = (cells["coorX"].max() - cells["coorX"].min()) / 2
    y_mid = (cells["coorY"].max() - cells["coorY"].min()) / 2
    ax.set(xlim=(x_mid - 350, x_mid + 350), ylim=(y_mid - 350, y_mid + 350))
    ax.set_axis_off()
    f.savefig(f"Results/Fig5/{filename}.svg", dpi=600, bbox_inches="tight")
    plt.close()

# --- Plot Template Patient ---
template_id = "MB-0882"
template_idx = patient_ids.index(template_id)
template_image_id = 50
template_cells = SC_d[SC_d["imageID"] == template_image_id].reset_index(drop=True)
template_adj = Cellular_graphs[template_idx][1]
print(f"Template {template_id} has {len(template_cells)} cells")
draw_graph(template_adj, template_cells, "fig5_a")

# --- Plot Similarity Curve ---
rep_ids = ["MB-0262", "MB-0356", "MB-0290"]
similarities = np.sort(Similarity_matrix[template_idx])[::-1]
rep_indices = [patient_ids.index(pid) for pid in rep_ids]
sorted_indices = np.argsort(Similarity_matrix[template_idx])[::-1]
rep_ranks = [list(sorted_indices).index(i) for i in rep_indices]

f, ax = plt.subplots(figsize=(8, 1.5))
ax.plot(similarities[:100], color="k")
ax.scatter(0, similarities[0], color="red", s=8)  # template
for rank in rep_ranks:
    ax.scatter(rank, similarities[rank], color="red", s=8)
    print(f"Representative: {patient_ids[sorted_indices[rank]]} | Similarity = {similarities[rank]:.4f} | Rank = {rank}")
ax.set(xlabel="Patient Rank", ylabel="Similarity")
f.savefig("Results/Fig5/fig5_c.svg", dpi=600, bbox_inches="tight")
plt.close()

# --- Plot Representative Patients ---
for i, pid in enumerate(rep_ids):
    image_id = SC_d[SC_d["patientID"] == pid]["imageID"].values[0]
    cells = SC_d[SC_d["imageID"] == image_id].reset_index(drop=True)
    adj = Cellular_graphs[patient_ids.index(pid)][1]
    print(f"{pid} (Image ID: {image_id}) has {len(cells)} cells")
    draw_graph(adj, cells, f"fig5_{chr(ord('d') + i)}")
