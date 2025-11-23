import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from utils import preprocess_Danenberg
sys.path.extend(["./..", "./../.."])
from bi_graph import BiGraph
from population_graph import Population_Graph


# --- Setup ---
sys.path.append("./..")
os.makedirs("Results", exist_ok=True)

# --- Data Load ---
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, _, survival_d, _ = preprocess_Danenberg(SC_d_raw, survival_d_raw)

# --- Model Fit ---
bigraph = BiGraph(k_patient_clustering=30, sample_frac=0.1, soft_wl_save_path="Fitted_swl/Danenberg/fitted_soft_wl_subtree_sample_0.1.pkl")
_, patient_subgroups = bigraph.fit_transform(SC_d, survival_data=survival_d)

# --- Subgroup Assignment ---
patient_ids = SC_d["patientID"].unique()
subgroup_ids = np.full(len(patient_ids), "Unclassified", dtype=object)
for subgroup in patient_subgroups:
    idx = np.isin(patient_ids, subgroup["patient_ids"])
    subgroup_ids[idx] = subgroup["subgroup_id"]

# --- Color Palette ---
color_palette = {"Unclassified": "white"}
color_palette.update({f"S{i+1}": sns.color_palette("tab10")[i] for i in range(10)})

# === Function: Plot Population Graph ===
def plot_population_graph(pop_graph, pos, colors, filename, white_nodes=False):
    fig = plt.figure(figsize=(5, 5), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    node_xyz = np.array([pos[v] for v in sorted(pop_graph)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in pop_graph.edges()])
    
    ax.scatter(
        *node_xyz.T,
        s=60,
        c='white' if white_nodes else [colors[i] for i in subgroup_ids],
        edgecolors="black",
        linewidths=1,
        alpha=1,
    )
    edge_alpha = [
        0.2 * pop_graph[u][v]["weight"] if pop_graph[u][v]["weight"] > 0.1 else 0
        for u, v in pop_graph.edges()
    ]
    for i, (u, v) in enumerate(pop_graph.edges()):
        ax.plot(*edge_xyz[i].T, alpha=edge_alpha[i], color="k")

    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.set_xlim(node_xyz[:, 0].min(), node_xyz[:, 0].max())
    ax.set_ylim(node_xyz[:, 1].min(), node_xyz[:, 1].max())
    ax.set_zlim(node_xyz[:, 2].min(), node_xyz[:, 2].max())
    fig.savefig(f"Results/{filename}.svg", dpi=600, bbox_inches="tight")
    plt.close()

# --- Generate Population Graph ---
pop_graph = Population_Graph(k_clustering=20).generate(bigraph.Similarity_matrix, patient_ids)
pos = nx.spring_layout(pop_graph, seed=3, k=1/np.sqrt(397)*10, iterations=100, dim=3)

# --- Plot Population Graphs ---
plot_population_graph(pop_graph, pos, color_palette, "population_graph", white_nodes=False)

# --- Print Subgroup Stats ---
for subgroup in patient_subgroups:
    print(f"Patient subgroup {subgroup['subgroup_id']}: "
          f"N = {len(subgroup['patient_ids'])}, "
          f"HR = {subgroup['hr']:.2f} ({subgroup['hr_lower']:.2f}-{subgroup['hr_upper']:.2f}), "
          f"p = {subgroup['p']:.2e}")

# --- Survival Data Prep ---
lengths = np.array([survival_d.loc[survival_d["patientID"] == pid, "time"].values[0] for pid in patient_ids])
statuses = np.array([survival_d.loc[survival_d["patientID"] == pid, "status"].values[0] for pid in patient_ids])

# === Function: Kaplan-Meier Plot ===
def plot_km(subgroups, filename, highlight=None):
    f, ax = plt.subplots(figsize=(5, 5))
    kmf = KaplanMeierFitter()

    for subgroup in subgroups:
        sid = subgroup["subgroup_id"]
        idx = (subgroup_ids == sid)
        label = f"{sid} (N={idx.sum()})"
        if highlight and sid in highlight:
            label = f"{sid}: {highlight[sid]} (N={idx.sum()})"
        kmf.fit(lengths[idx], statuses[idx], label=label)
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=color_palette[sid],
            show_censors=True,
            linewidth=2,
            censor_styles={"ms": 5, "marker": "|"},
        )
    
    ax.set_xlabel("Time (Month)", fontsize=14)
    ax.set_ylabel("Cumulative Survival", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=13, ncol=2 if not highlight else 1, loc="lower left")
    sns.despine()
    f.savefig(f"Results/{filename}.svg", dpi=600, bbox_inches="tight")
    plt.close()

# --- All Subgroups KM ---
logrank = multivariate_logrank_test(lengths[subgroup_ids != "Unclassified"],
                                     subgroup_ids[subgroup_ids != "Unclassified"],
                                     statuses[subgroup_ids != "Unclassified"])
p_val = logrank.p_value
plot_km(patient_subgroups, "survivial_curves")

# === Plot Hazard Ratios ===
f, ax = plt.subplots(2, 1, figsize=(8, 3), height_ratios=[5, 1], sharex=True)
f.subplots_adjust(hspace=0)

ax[0].hlines(1, -1, len(patient_subgroups), color="k", linestyle="--")
xticklabels, xtickcolors, N = [], [], []

for i, subgroup in enumerate(patient_subgroups):
    sid, hr, hr_lb, hr_ub, p = (
        subgroup["subgroup_id"],
        subgroup["hr"],
        subgroup["hr_lower"],
        subgroup["hr_upper"],
        subgroup["p"]
    )
    ax[0].plot([i, i], [hr_lb, hr_ub], color=color_palette[sid], linewidth=2)
    ax[0].scatter([i], [hr], color=color_palette[sid], s=120)
    ax[0].scatter([i], [hr_lb], color=color_palette[sid], s=60, marker="_")
    ax[0].scatter([i], [hr_ub], color=color_palette[sid], s=60, marker="_")
    
    xticklabels.append(sid)
    xtickcolors.append("k" if p < 0.05 else "gray")
    N.append((subgroup_ids == sid).sum())

ax[0].set_xticks(range(len(patient_subgroups)))
ax[0].set_xticklabels(xticklabels)
for tick, color in zip(ax[1].get_xticklabels(), xtickcolors):
    tick.set_color(color)
ax[0].set_yscale("log")
ax[0].set_xlabel("Patient Subgroups")
ax[0].set_ylabel("Hazard ratio")

sns.barplot(x="subgroup_id", y="N", data=pd.DataFrame({"subgroup_id": xticklabels, "N": N}),
            palette=color_palette, ax=ax[1])
ax[1].invert_yaxis()
ax[1].set_ylabel("Size")
ax[1].set_xlabel("")
f.savefig("Results/hazard_ratios.svg", dpi=600, bbox_inches="tight")
