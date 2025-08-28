import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter
from utils import preprocess_Danenberg, preprocess_Jackson
from definitions import (
    color_palette_Bigraph,
    get_paired_markers,
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

from sklearn.neighbors import NearestNeighbors
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

patient_ids_iv = list(SC_iv["patientID"].unique())
subgroup_ids_iv = np.zeros(len(patient_ids_iv), dtype=object)
subgroup_ids_iv[:] = "Unclassified"
for i in range(len(patient_subgroups_iv)):
    subgroup = patient_subgroups_iv[i]
    subgroup_id = subgroup["subgroup_id"]
    patient_ids = subgroup["patient_ids"]
    subgroup_ids_iv[np.isin(patient_ids_iv, patient_ids)] = subgroup_id


pos = nx.spring_layout(
    population_graph_iv,
    seed=100,
    k=1 / (np.sqrt(397)) * 10,
    iterations=100,
    dim=3,
)
fig = plt.figure(figsize=(5, 5), tight_layout=True)
ax = fig.add_subplot(111, projection="3d")
node_xyz = np.array([pos[v] for v in sorted(population_graph_iv)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in population_graph_iv.edges()])
ax.scatter(
    *node_xyz.T,
    s=60,
    c=[color_palette_Bigraph[i[:-1]] for i in subgroup_ids_iv],
    edgecolors="black",
    linewidths=1,
    alpha=1
)
edge_list = list(population_graph_iv.edges())
edge_alpha = [
    (
        0.2 * population_graph_iv[u][v]["weight"]
        if population_graph_iv[u][v]["weight"] > 0.1
        else 0
    )
    for u, v in edge_list
]
for i in range(len(edge_list)):
    u, v = edge_list[i]
    ax.plot(*edge_xyz[i].T, alpha=edge_alpha[i], color="k")

ax.set(
    xlim=(np.min(node_xyz[:, 0]), np.max(node_xyz[:, 0])),
    ylim=(np.min(node_xyz[:, 1]), np.max(node_xyz[:, 1])),
    zlim=(np.min(node_xyz[:, 2]), np.max(node_xyz[:, 2])),
)
handles = []
if np.sum(subgroup_ids_iv == "Unclassified") > 0:
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="white",
            label="Unclassified (N = {})".format(
                np.sum(subgroup_ids_iv == "Unclassified")
            ),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )
for subgroup in patient_subgroups_iv:
    subgroup_id = subgroup["subgroup_id"]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markerfacecolor=color_palette_Bigraph[subgroup_id[:-1]],
            label="{} (N = {})".format(
                subgroup_id, np.sum(subgroup_ids_iv == subgroup_id)
            ),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )

# ax.legend(handles=handles, fontsize=12, ncols=2)
ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
plt.savefig("Results/fig13_a.png", dpi=600, bbox_inches="tight")
plt.savefig("Results/fig13_a.svg", dpi=300, bbox_inches="tight")


for i in range(len(patient_subgroups_iv)):
    print(
        "Patient subgroup {}: N = {}, HR = {:.2f} ({:.2f}-{:.2f}), p = {:.2e}".format(
            patient_subgroups_iv[i]["subgroup_id"],
            len(patient_subgroups_iv[i]["patient_ids"]),
            patient_subgroups_iv[i]["hr"],
            patient_subgroups_iv[i]["hr_lower"],
            patient_subgroups_iv[i]["hr_upper"],
            patient_subgroups_iv[i]["p"],
        )
    )

lengths_iv = [
    survival_iv.loc[survival_iv["patientID"] == i, "time"].values[0]
    for i in patient_ids_iv
]
statuses_iv = [
    survival_iv.loc[survival_iv["patientID"] == i, "status"].values[0]
    for i in patient_ids_iv
]
kmf = KaplanMeierFitter()
f, ax = plt.subplots(figsize=(5.5, 5.5))
for subgroup in patient_subgroups_iv:
    subgroup_id = subgroup["subgroup_id"]
    length_A, event_observed_A = (
        np.array(lengths_iv)[subgroup_ids_iv == subgroup_id],
        np.array(statuses_iv)[subgroup_ids_iv == subgroup_id],
    )
    label = "{} (N={})".format(
        subgroup["subgroup_id"], np.sum(subgroup_ids_iv == subgroup_id)
    )
    kmf.fit(length_A, event_observed_A, label=label)
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        color=color_palette_Bigraph[subgroup_id[:-1]],
        show_censors=True,
        linewidth=2,
        censor_styles={"ms": 5, "marker": "|"},
    )
log_rank_test = multivariate_logrank_test(
    np.array(lengths_iv)[subgroup_ids_iv != 0],
    np.array(subgroup_ids_iv)[subgroup_ids_iv != 0],
    np.array(statuses_iv)[subgroup_ids_iv != 0],
)
p_value = log_rank_test.p_value
ax.legend(ncol=2, fontsize=14)
ax.text(
    x=0.3,
    y=0.95,
    s="p-value = {:.5f}".format(p_value),
    fontsize=16,
    transform=ax.transAxes,
)
ax.set_xlabel("Time (Month)", fontsize=16)
ax.set_ylabel("Cumulative Survival", fontsize=16)
ax.set(
    ylim=(-0.05, 1.1),
)
sns.despine()
plt.savefig("Results/fig13_b.jpg", dpi=300, bbox_inches="tight")
plt.savefig("Results/fig13_b.svg", dpi=300, bbox_inches="tight")


# Plot hazard ratio
f, ax = plt.subplots(2, 1, height_ratios=[5, 1], figsize=(8, 3), sharex=True)
f.subplots_adjust(hspace=0)
ax[0].hlines(1, -1, len(patient_subgroups_iv), color="k", linestyle="--")
N, xticklabels, xtickcolors = [], [], []
for i in range(len(patient_subgroups_iv)):
    subgroup = patient_subgroups_iv[i]
    subgroup_id = subgroup["subgroup_id"]
    hr, hr_lb, hr_ub, p = (
        subgroup["hr"],
        subgroup["hr_lower"],
        subgroup["hr_upper"],
        subgroup["p"],
    )
    ax[0].plot(
        [i, i],
        [hr_lb, hr_ub],
        color=color_palette_Bigraph[subgroup_id[:-1]],
        linewidth=2,
    )
    ax[0].scatter([i], [hr], color=color_palette_Bigraph[subgroup_id[:-1]], s=120)
    ax[0].scatter(
        [i], [hr_lb], color=color_palette_Bigraph[subgroup_id[:-1]], s=60, marker="_"
    )
    ax[0].scatter(
        [i], [hr_ub], color=color_palette_Bigraph[subgroup_id[:-1]], s=60, marker="_"
    )
    N.append(np.sum(subgroup_ids_iv == subgroup_id))
    xticklabels.append("{}".format(subgroup_id))
    if p < 0.05:
        xtickcolors.append("k")
    else:
        xtickcolors.append("grey")
ax[0].set_xticks(range(len(patient_subgroups_iv)))
ax[0].set_xticklabels(xticklabels)
for xtick, color in zip(ax[1].get_xticklabels(), xtickcolors):
    xtick.set_color(color)
ax[0].set_xlabel("Patient Subgroups")
ax[0].set_ylabel("Hazard ratio")
ax[0].set_yscale("log")
DF = pd.DataFrame({"N": N, "subgroup_id": xticklabels})
g = sns.barplot(
    data=DF,
    x="subgroup_id",
    y="N",
    palette={key + "'": value for key, value in color_palette_Bigraph.items()},
    ax=ax[1],
)
g.invert_yaxis()
ax[1].set_ylabel("Size")
ax[1].set_xlabel("")
plt.show()
f.savefig("Results/fig13_c.jpg", dpi=300, bbox_inches="tight")
f.savefig("Results/fig13_c.SVG", dpi=300, bbox_inches="tight")

patient_ids_ev = list(SC_ev["patientID"].unique())
subgroup_ids_ev = np.zeros(len(patient_ids_ev), dtype=object)
subgroup_ids_ev[:] = "Unclassified"
for i in range(len(patient_subgroups_ev)):
    subgroup = patient_subgroups_ev[i]
    subgroup_id = subgroup["subgroup_id"]
    patient_ids = subgroup["patient_ids"]
    subgroup_ids_ev[np.isin(patient_ids_ev, patient_ids)] = subgroup_id

pos = nx.spring_layout(
    population_graph_ev,
    seed=1,
    k=1 / (np.sqrt(397)) * 10,
    iterations=100,
    dim=3,
)
fig = plt.figure(figsize=(5, 5), tight_layout=True)
ax = fig.add_subplot(111, projection="3d")
node_xyz = np.array([pos[v] for v in sorted(population_graph_ev)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in population_graph_ev.edges()])
ax.scatter(
    *node_xyz.T,
    s=60,
    c=[color_palette_Bigraph[i[:-1]] for i in subgroup_ids_ev],
    edgecolors="black",
    linewidths=1,
    alpha=1
)
edge_list = list(population_graph_ev.edges())
edge_alpha = [
    (
        0.2 * population_graph_ev[u][v]["weight"]
        if population_graph_ev[u][v]["weight"] > 0.1
        else 0
    )
    for u, v in edge_list
]
for i in range(len(edge_list)):
    u, v = edge_list[i]
    ax.plot(*edge_xyz[i].T, alpha=edge_alpha[i], color="k")

ax.set(
    xlim=(np.min(node_xyz[:, 0]), np.max(node_xyz[:, 0])),
    ylim=(np.min(node_xyz[:, 1]), np.max(node_xyz[:, 1])),
    zlim=(np.min(node_xyz[:, 2]), np.max(node_xyz[:, 2])),
)
handles = []
if np.sum(subgroup_ids_ev == "Unclassified") > 0:
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="white",
            label="Unclassified (N = {})".format(
                np.sum(subgroup_ids_ev == "Unclassified")
            ),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )
for subgroup in patient_subgroups_ev:
    subgroup_id = subgroup["subgroup_id"]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markerfacecolor=color_palette_Bigraph[subgroup_id[:-1]],
            label="{} (N = {})".format(
                subgroup_id, np.sum(subgroup_ids_ev == subgroup_id)
            ),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )

# ax.legend(handles=handles, fontsize=12, ncols=2, loc="lower left")
ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
plt.savefig("Results/fig13_d.png", dpi=600, bbox_inches="tight")
plt.savefig("Results/fig13_d.svg", dpi=300, bbox_inches="tight")


for i in range(len(patient_subgroups_ev)):
    print(
        "Patient subgroup {}: N = {}, HR = {:.2f} ({:.2f}-{:.2f}), p = {:.2e}".format(
            patient_subgroups_ev[i]["subgroup_id"],
            len(patient_subgroups_ev[i]["patient_ids"]),
            patient_subgroups_ev[i]["hr"],
            patient_subgroups_ev[i]["hr_lower"],
            patient_subgroups_ev[i]["hr_upper"],
            patient_subgroups_ev[i]["p"],
        )
    )

lengths_ev = [
    survival_ev.loc[survival_ev["patientID"] == i, "time"].values[0]
    for i in patient_ids_ev
]
statuses_ev = [
    survival_ev.loc[survival_ev["patientID"] == i, "status"].values[0]
    for i in patient_ids_ev
]
kmf = KaplanMeierFitter()
f, ax = plt.subplots(figsize=(5.5, 5.5))
for subgroup in patient_subgroups_ev:
    subgroup_id = subgroup["subgroup_id"]
    length_A, event_observed_A = (
        np.array(lengths_ev)[subgroup_ids_ev == subgroup_id],
        np.array(statuses_ev)[subgroup_ids_ev == subgroup_id],
    )
    label = "{} (N={})".format(
        subgroup["subgroup_id"], np.sum(subgroup_ids_ev == subgroup_id)
    )
    kmf.fit(length_A, event_observed_A, label=label)
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        color=color_palette_Bigraph[subgroup_id[:-1]],
        show_censors=True,
        linewidth=2,
        censor_styles={"ms": 5, "marker": "|"},
    )
log_rank_test = multivariate_logrank_test(
    np.array(lengths_ev)[subgroup_ids_ev != 0],
    np.array(subgroup_ids_ev)[subgroup_ids_ev != 0],
    np.array(statuses_ev)[subgroup_ids_ev != 0],
)
p_value = log_rank_test.p_value
ax.legend(ncol=2, fontsize=14, loc="lower left")
ax.text(
    x=0.3,
    y=0.95,
    s="p-value = {:.5f}".format(p_value),
    fontsize=16,
    transform=ax.transAxes,
)
ax.set_xlabel("Time (Month)", fontsize=16)
ax.set_ylabel("Cumulative Survival", fontsize=16)
ax.set(
    ylim=(-0.15, 1.15),
)
sns.despine()
plt.savefig("Results/fig13_e.jpg", dpi=300, bbox_inches="tight")
plt.savefig("Results/fig13_e.svg", dpi=300, bbox_inches="tight")


# Plot hazard ratio
f, ax = plt.subplots(2, 1, height_ratios=[5, 1], figsize=(8, 3), sharex=True)
f.subplots_adjust(hspace=0)
ax[0].hlines(1, -1, len(patient_subgroups_ev), color="k", linestyle="--")
N, xticklabels, xtickcolors = [], [], []
for i in range(len(patient_subgroups_ev)):
    subgroup = patient_subgroups_ev[i]
    subgroup_id = subgroup["subgroup_id"]
    hr, hr_lb, hr_ub, p = (
        subgroup["hr"],
        subgroup["hr_lower"],
        subgroup["hr_upper"],
        subgroup["p"],
    )
    ax[0].plot(
        [i, i],
        [hr_lb, hr_ub],
        color=color_palette_Bigraph[subgroup_id[:-1]],
        linewidth=2,
    )
    ax[0].scatter([i], [hr], color=color_palette_Bigraph[subgroup_id[:-1]], s=120)
    ax[0].scatter(
        [i], [hr_lb], color=color_palette_Bigraph[subgroup_id[:-1]], s=60, marker="_"
    )
    ax[0].scatter(
        [i], [hr_ub], color=color_palette_Bigraph[subgroup_id[:-1]], s=60, marker="_"
    )
    N.append(np.sum(subgroup_ids_ev == subgroup_id))
    xticklabels.append("{}".format(subgroup_id))
    if p < 0.05:
        xtickcolors.append("k")
    else:
        xtickcolors.append("grey")
ax[0].set_xticks(range(len(patient_subgroups_ev)))
ax[0].set_xticklabels(xticklabels)
for xtick, color in zip(ax[1].get_xticklabels(), xtickcolors):
    xtick.set_color(color)
ax[0].set_xlabel("Patient Subgroups")
ax[0].set_ylabel("Hazard ratio")
ax[0].set_yscale("log")
DF = pd.DataFrame({"N": N, "subgroup_id": xticklabels})
g = sns.barplot(
    data=DF,
    x="subgroup_id",
    y="N",
    palette={key + "'": value for key, value in color_palette_Bigraph.items()},
    ax=ax[1],
)
g.invert_yaxis()
ax[1].set_ylabel("Size")
ax[1].set_xlabel("")
plt.show()
f.savefig("Results/fig13_f.jpg", dpi=300, bbox_inches="tight")
f.savefig("Results/fig13_f.svg", dpi=300, bbox_inches="tight")


