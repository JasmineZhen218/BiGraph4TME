import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_Danenberg, preprocess_Jackson
from sklearn.neighbors import NearestNeighbors
from definitions import (
    Cell_types_displayed_Danenberg,
    get_paired_markers
)
sys.path.append("./..")
from bi_graph import BiGraph

SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
SC_ev_raw = pd.read_csv("Datasets/Jackson_et_al/cells.csv")
survival_ev_raw = pd.read_csv("Datasets/Jackson_et_al/clinical.csv")
SC_ev, survival_ev = preprocess_Jackson(SC_ev_raw, survival_ev_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)
population_graph_iv, patient_subgroups_iv, histograms_iv, Signature_iv = bigraph_.transform(
    SC_iv, survival_data=survival_iv
)

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
population_graph_ev, patient_subgroups_ev, histograms_ev, Signature_ev = bigraph_.transform(
    SC_ev, survival_data=survival_ev
)


Signatures = bigraph_.fitted_soft_wl_subtree.Signatures
threshold = 0.5  # does not impact the downstream analysis, only imapct the presentation of the signature map
tumor_niche = np.where(
    (np.sum(Signatures[:, :16] > threshold, axis=1) > 0)
    & (np.sum(Signatures[:, 16:] > threshold, axis=1) == 0)
)[0]
immune_niche = np.where(
    (np.sum(Signatures[:, :16] > threshold, axis=1) == 0)
    & (np.sum(Signatures[:, 16:27] > threshold, axis=1) > 0)
    & (np.sum(Signatures[:, 27:] > threshold, axis=1) == 0)
)[0]
stromal_niche = np.where(
    (np.sum(Signatures[:, :16] > threshold, axis=1) == 0)
    & (np.sum(Signatures[:, 16:27] > threshold, axis=1) == 0)
    & (np.sum(Signatures[:, 27:] > threshold, axis=1) > 0)
)[0]
interacting_niche = [
    i
    for i in range(Signatures.shape[0])
    if i not in np.concatenate([tumor_niche, immune_niche, stromal_niche])
]
print("There are {} identified TME patterns.".format(Signatures.shape[0]))
print(
    "There are {} tumor niches, {} immune niches, {} stromal niches, and {} interacting niches.".format(
        len(tumor_niche), len(immune_niche), len(stromal_niche), len(interacting_niche)
    )
)
tme_pattern_orders = np.concatenate(
    [tumor_niche, immune_niche, stromal_niche, interacting_niche]
)
f, ax = plt.subplots(
    1,
    2,
    figsize=(12, 10),
    tight_layout=True,
    gridspec_kw={"width_ratios": [10, 0.3]},
)

sns.heatmap(
    Signature_iv[tme_pattern_orders, :],
    ax=ax[0],
    cmap="rocket_r",
    linewidth=0.005,
    cbar=True,
    cbar_ax=ax[1],
    edgecolor="black",
    vmax=np.percentile(Signatures, 99),
    vmin=np.percentile(Signatures, 5),
)

ax[0].get_yaxis().set_visible(False)
ax[0].set_title("Signature map", fontsize=14)
ax[0].set_xticklabels(
    Cell_types_displayed_Danenberg, rotation=90, fontsize=12, fontweight="bold"
)
ax[0].set_xlabel("Cell Phenotypes", fontsize=14)
xtickcolors = ["cornflowerblue"] * 16 + ["darkorange"] * 11 + ["seagreen"] * 5
for xtick, color in zip(ax[0].get_xticklabels(), xtickcolors):
    xtick.set_color(color)
ax[1].get_xaxis().set_visible(False)
f.savefig("Results/figA6_b.png", dpi=300, bbox_inches="tight")
f.savefig("Results/figA6_b.svg", dpi=300, bbox_inches="tight")

f, ax = plt.subplots(
    1,
    2,
    figsize=(12, 10),
    tight_layout=True,
    gridspec_kw={"width_ratios": [10, 0.3]},
)

sns.heatmap(
    Signature_ev[tme_pattern_orders, :],
    ax=ax[0],
    cmap="rocket_r",
    linewidth=0.005,
    cbar=True,
    cbar_ax=ax[1],
    edgecolor="black",
    vmax=np.percentile(Signatures, 99),
    vmin=np.percentile(Signatures, 5),
)

ax[0].get_yaxis().set_visible(False)
ax[0].set_title("Signature map", fontsize=14)
ax[0].set_xticklabels(
    Cell_types_displayed_Danenberg, rotation=90, fontsize=12, fontweight="bold"
)
ax[0].set_xlabel("Cell Phenotypes", fontsize=14)
xtickcolors = ["cornflowerblue"] * 16 + ["darkorange"] * 11 + ["seagreen"] * 5
for xtick, color in zip(ax[0].get_xticklabels(), xtickcolors):
    xtick.set_color(color)
ax[1].get_xaxis().set_visible(False)
f.savefig("Results/figA6_c.png", dpi=300, bbox_inches="tight")
f.savefig("Results/figA6_c.svg", dpi=300, bbox_inches="tight")