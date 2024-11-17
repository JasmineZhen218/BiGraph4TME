import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from utils import reverse_dict, preprocess_Danenberg, preprocess_Jackson
from definitions import (
    get_node_id,
    Cell_types_displayed_Danenberg,
    Cell_types_displayed_Jackson,
    get_paired_markers,
)

SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
SC_ev_raw = pd.read_csv("Datasets/Jackson_et_al/cells.csv")
survival_ev_raw = pd.read_csv("Datasets/Jackson_et_al/clinical.csv")
SC_ev, survival_ev = preprocess_Jackson(SC_ev_raw, survival_ev_raw)
def map_cell_types(SC_d, SC_ev, paired_proteins):
    cell_types_d = SC_d["celltypeID"].values
    protein_expression_d = SC_d[[i[0] for i in paired_proteins]].values
    # normalize protein expression
    protein_expression_d = (
        protein_expression_d - np.mean(protein_expression_d, axis=0)
    ) / np.std(protein_expression_d, axis=0)
    protein_expression_ev = SC_ev[[i[1] for i in paired_proteins]].values
    # normalize protein expression
    protein_expression_ev = (
        protein_expression_ev - np.mean(protein_expression_ev, axis=0)
    ) / np.std(protein_expression_ev, axis=0)
    centroids = np.zeros((len(np.unique(cell_types_d)), len(paired_proteins)))
    for i in range(len(np.unique(cell_types_d))):
        centroids[i] = np.mean(protein_expression_d[cell_types_d == i], axis=0)
    cell_types_ev_hat = np.zeros(len(protein_expression_ev), dtype=int)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(centroids)
    # to avoid memory error, we split the data into chunks
    n_chunks = 100
    chunk_size = len(protein_expression_ev) // n_chunks
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        _, indices = neigh.kneighbors(protein_expression_ev[start:end])
        cell_types_ev_hat[start:end] = indices.flatten()
    # the last chunk
    _, indices = neigh.kneighbors(protein_expression_ev[end:])
    cell_types_ev_hat[end:] = indices.flatten()
    SC_ev["celltypeID_original"] = SC_ev["celltypeID"]
    SC_ev["celltypeID"] = list(cell_types_ev_hat)
    return SC_ev
SC_ev = map_cell_types(
    SC_d, SC_ev, get_paired_markers(source="Danenberg", target="Jackson")
)

Label_Query = SC_ev["celltypeID_original"].values
Label_Query_alignment = SC_ev["celltypeID"].values
Matching_heatmap = np.zeros(
    (len(np.unique(Label_Query)), len(np.unique(Label_Query_alignment)))
)
for i in range(len(Cell_types_displayed_Jackson)):
    for j in range(len(Cell_types_displayed_Danenberg)):
        id_i = get_node_id("Jackson", "CellType")[Cell_types_displayed_Jackson[i]]
        id_j = j
        a = list(np.where(Label_Query == id_i)[0])
        b = list(np.where(Label_Query_alignment == id_j)[0])
        IoU = len(set(a).intersection(set(b))) / len(set(a).union(set(b)))
        Matching_heatmap[i, j] = IoU

Cell_types_original = list(
    map(
        reverse_dict(get_node_id("Jackson", "CellType")).get,
        range(np.unique(Label_Query).shape[0]),
    )
)
Cell_types_aligned = list(
    map(
        reverse_dict(get_node_id("Danenberg", "CellType")).get,
        range(np.unique(Label_Query_alignment).shape[0]),
    )
)
f, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
sns.heatmap(
    Matching_heatmap,
    ax=ax,
    xticklabels=Cell_types_displayed_Danenberg,
    yticklabels=Cell_types_displayed_Jackson,
    cmap="RdBu_r",
    vmin=0,
    # vmax=0.3,
    linewidths=0.5,
)
ax.set_ylabel("Cell types in external validation set", fontsize=14)
ax.set_xlabel("Cell types in discovery set", fontsize=14)
ax.set_title(
    "Cell type alignment between Discovery and External validation set-1 (Jackson et al.)", fontsize=14
)
ax.set_yticklabels(Cell_types_displayed_Jackson, fontsize=12, fontweight="bold")
ytickcolors = ["cornflowerblue"] * 13 + ["darkorange"] * 6 + ["forestgreen"] * 7
for ytick, color in zip(ax.get_yticklabels(), ytickcolors):
    ytick.set_color(color)
ax.set_xticklabels(Cell_types_displayed_Danenberg, fontsize=12, fontweight="bold")
xtickcolors = ["cornflowerblue"] * 16 + ["darkorange"] * 11 + ["forestgreen"] * 5
for xtick, color in zip(ax.get_xticklabels(), xtickcolors):
    xtick.set_color(color)
f.savefig("Results/figA5_a.png", dpi=300, bbox_inches="tight")


