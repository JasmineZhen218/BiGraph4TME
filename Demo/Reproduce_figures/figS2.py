import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from utils import preprocess_Danenberg
sys.path.append("./..")
from bi_graph import BiGraph
from population_graph import Population_Graph


SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
# bigraph_ = BiGraph(k_patient_clustering=20)
# population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
#     SC_d, survival_data=survival_d
# )

# patient_ids_discovery = list(SC_d["patientID"].unique())
# subgroup_ids_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
# subgroup_ids_discovery[:] = "Unclassified"
# for i in range(len(patient_subgroups_discovery)):
#     subgroup = patient_subgroups_discovery[i]
#     subgroup_id = subgroup["subgroup_id"]
#     patient_ids = subgroup["patient_ids"]
#     subgroup_ids_discovery[np.isin(patient_ids_discovery, patient_ids)] = subgroup_id

# color_palette_Bigraph = {
#     "Unclassified": "white",
#     "S1": sns.color_palette("tab10")[0],
#     "S2": sns.color_palette("tab10")[1],
#     "S3": sns.color_palette("tab10")[2],
#     "S4": sns.color_palette("tab10")[3],
#     "S5": sns.color_palette("tab10")[4],
#     "S6": sns.color_palette("tab10")[5],
#     "S7": sns.color_palette("tab10")[6],
#     "S8": sns.color_palette("tab10")[7],
#     "S9": sns.color_palette("tab10")[8],
#     "S10": sns.color_palette("tab10")[9],
# }

# # For visualization purpose, we make nodes distant from each other
# population_graph_for_visualization = Population_Graph(k_clustering=20).generate(
#     bigraph_.Similarity_matrix, patient_ids_discovery
# )  # generate population graph
# pos = nx.spring_layout(
#     population_graph_for_visualization,
#     seed=3,
#     k=1 / (np.sqrt(397)) * 10,
#     iterations=100,
#     dim=3,
# )
# fig = plt.figure(figsize=(5, 5), tight_layout=True)
# ax = fig.add_subplot(111, projection="3d")
# node_xyz = np.array([pos[v] for v in sorted(population_graph_for_visualization)])
# edge_xyz = np.array(
#     [(pos[u], pos[v]) for u, v in population_graph_for_visualization.edges()]
# )
# ax.scatter(
#     *node_xyz.T,
#     s=60,
#     c=[color_palette_Bigraph[i] for i in subgroup_ids_discovery],
#     edgecolors="black",
#     linewidths=1,
#     alpha=1
# )
# edge_list = list(population_graph_for_visualization.edges())
# edge_alpha = [
#     (
#         0.2 * population_graph_for_visualization[u][v]["weight"]
#         if population_graph_for_visualization[u][v]["weight"] > 0.1
#         else 0
#     )
#     for u, v in edge_list
# ]
# for i in range(len(edge_list)):
#     u, v = edge_list[i]
#     ax.plot(*edge_xyz[i].T, alpha=edge_alpha[i], color="k")

# ax.set(
#     xlim=(np.min(node_xyz[:, 0]), np.max(node_xyz[:, 0])),
#     ylim=(np.min(node_xyz[:, 1]), np.max(node_xyz[:, 1])),
#     zlim=(np.min(node_xyz[:, 2]), np.max(node_xyz[:, 2])),
# )
# handles = []
# if np.sum(subgroup_ids_discovery == "Unclassified") > 0:
#     handles.append(
#         Line2D(
#             [0],
#             [0],
#             marker="o",
#             color="white",
#             markerfacecolor="white",
#             label="Unclassified (N = {})".format(
#                 np.sum(subgroup_ids_discovery == "Unclassified")
#             ),
#             markeredgecolor="black",
#             markeredgewidth=1,
#             markersize=8,
#         )
#     )
# for subgroup in patient_subgroups_discovery:
#     subgroup_id = subgroup["subgroup_id"]
#     handles.append(
#         Line2D(
#             [0],
#             [0],
#             marker="o",
#             color="grey",
#             markerfacecolor=color_palette_Bigraph[subgroup_id],
#             label="{} (N = {})".format(
#                 subgroup_id, np.sum(subgroup_ids_discovery == subgroup_id)
#             ),
#             markeredgecolor="black",
#             markeredgewidth=1,
#             markersize=8,
#         )
#     )

# # ax.legend(handles=handles, fontsize=10, ncols=2)
# ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
# fig.savefig("Results/figS2_a_1.png", dpi=600, bbox_inches="tight")
# # fig.savefig("Results/figS2_a_1.svg", dpi=600, bbox_inches="tight")
# f, ax = plt.subplots(figsize=(5, 5))
# ax.legend(handles=handles, fontsize=10, ncols=2)
# f.savefig("Results/figS2_a_1_legend.svg", dpi=600, bbox_inches="tight")



# for i in range(len(patient_subgroups_discovery)):
#     print(
#         "Patient subgroup {}: N = {}, HR = {:.2f} ({:.2f}-{:.2f}), p = {:.2e}".format(
#             patient_subgroups_discovery[i]["subgroup_id"],
#             len(patient_subgroups_discovery[i]["patient_ids"]),
#             patient_subgroups_discovery[i]["hr"],
#             patient_subgroups_discovery[i]["hr_lower"],
#             patient_subgroups_discovery[i]["hr_upper"],
#             patient_subgroups_discovery[i]["p"],
#         )
#     )

# lengths_discovery = [
#     survival_d.loc[survival_d["patientID"] == i, "time"].values[0]
#     for i in patient_ids_discovery
# ]
# statuses_discovery = [
#     (survival_d.loc[survival_d["patientID"] == i, "status"].values[0])
#     for i in patient_ids_discovery
# ]
# kmf = KaplanMeierFitter()
# f, ax = plt.subplots(figsize=(5, 5))
# for subgroup in patient_subgroups_discovery:
#     subgroup_id = subgroup["subgroup_id"]
#     length_A, event_observed_A = (
#         np.array(lengths_discovery)[subgroup_ids_discovery == subgroup_id],
#         np.array(statuses_discovery)[subgroup_ids_discovery == subgroup_id],
#     )
#     label = "{} (N={})".format(
#         subgroup["subgroup_id"], np.sum(subgroup_ids_discovery == subgroup_id)
#     )
#     kmf.fit(length_A, event_observed_A, label=label)
#     kmf.plot_survival_function(
#         ax=ax,
#         ci_show=False,
#         color=color_palette_Bigraph[subgroup_id],
#         show_censors=True,
#         linewidth=2,
#         censor_styles={"ms": 5, "marker": "|"},
#     )
# log_rank_test = multivariate_logrank_test(
#     np.array(lengths_discovery)[subgroup_ids_discovery != 0],
#     np.array(subgroup_ids_discovery)[subgroup_ids_discovery != 0],
#     np.array(statuses_discovery)[subgroup_ids_discovery != 0],
# )
# p_value = log_rank_test.p_value
# ax.legend(ncol=2, fontsize=13)
# ax.text(
#     x=0.3,
#     y=0.95,
#     s="p-value = {:.5f}".format(p_value),
#     fontsize=14,
#     transform=ax.transAxes,
# )
# ax.set_xlabel("Time (Month)", fontsize=14)
# ax.set_ylabel("Cumulative Survival", fontsize=14)
# ax.set(
#     ylim=(-0.05, 1.05),
# )
# sns.despine()
# f.savefig("Results/figS2_a_2.png", dpi=600, bbox_inches="tight")
# f.savefig("Results/figS2_a_2.svg", dpi=600, bbox_inches="tight")




bigraph_ = BiGraph(k_patient_clustering=40)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)

patient_ids_discovery = list(SC_d["patientID"].unique())
subgroup_ids_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
subgroup_ids_discovery[:] = "Unclassified"
for i in range(len(patient_subgroups_discovery)):
    subgroup = patient_subgroups_discovery[i]
    subgroup_id = subgroup["subgroup_id"]
    patient_ids = subgroup["patient_ids"]
    subgroup_ids_discovery[np.isin(patient_ids_discovery, patient_ids)] = subgroup_id

color_palette_Bigraph = {
    "Unclassified": "white",
    "S1": sns.color_palette("tab10")[0],
    "S2": sns.color_palette("tab10")[1],
    "S3": sns.color_palette("tab10")[2],
    "S4": sns.color_palette("tab10")[3],
    "S5": sns.color_palette("tab10")[4],
    "S6": sns.color_palette("tab10")[5],
    "S7": sns.color_palette("tab10")[6],
    "S8": sns.color_palette("tab10")[7],
    "S9": sns.color_palette("tab10")[8],
    "S10": sns.color_palette("tab10")[9],
}

# For visualization purpose, we make nodes distant from each other
population_graph_for_visualization = Population_Graph(k_clustering=20).generate(
    bigraph_.Similarity_matrix, patient_ids_discovery
)  # generate population graph
pos = nx.spring_layout(
    population_graph_for_visualization,
    seed=3,
    k=1 / (np.sqrt(397)) * 10,
    iterations=100,
    dim=3,
)
fig = plt.figure(figsize=(5, 5), tight_layout=True)
ax = fig.add_subplot(111, projection="3d")
node_xyz = np.array([pos[v] for v in sorted(population_graph_for_visualization)])
edge_xyz = np.array(
    [(pos[u], pos[v]) for u, v in population_graph_for_visualization.edges()]
)
ax.scatter(
    *node_xyz.T,
    s=60,
    c=[color_palette_Bigraph[i] for i in subgroup_ids_discovery],
    edgecolors="black",
    linewidths=1,
    alpha=1
)
edge_list = list(population_graph_for_visualization.edges())
edge_alpha = [
    (
        0.2 * population_graph_for_visualization[u][v]["weight"]
        if population_graph_for_visualization[u][v]["weight"] > 0.1
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
if np.sum(subgroup_ids_discovery == "Unclassified") > 0:
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="white",
            label="Unclassified (N = {})".format(
                np.sum(subgroup_ids_discovery == "Unclassified")
            ),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )
for subgroup in patient_subgroups_discovery:
    subgroup_id = subgroup["subgroup_id"]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markerfacecolor=color_palette_Bigraph[subgroup_id],
            label="{} (N = {})".format(
                subgroup_id, np.sum(subgroup_ids_discovery == subgroup_id)
            ),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )

# ax.legend(handles=handles, fontsize=10, ncols=2)
ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
fig.savefig("Results/figS2_c_1.png", dpi=600, bbox_inches="tight")
# fig.savefig("Results/figS2_a_1.svg", dpi=600, bbox_inches="tight")
f, ax = plt.subplots(figsize=(5, 5))
ax.legend(handles=handles, fontsize=10, ncols=2)
f.savefig("Results/figS2_c_1_legend.svg", dpi=600, bbox_inches="tight")



for i in range(len(patient_subgroups_discovery)):
    print(
        "Patient subgroup {}: N = {}, HR = {:.2f} ({:.2f}-{:.2f}), p = {:.2e}".format(
            patient_subgroups_discovery[i]["subgroup_id"],
            len(patient_subgroups_discovery[i]["patient_ids"]),
            patient_subgroups_discovery[i]["hr"],
            patient_subgroups_discovery[i]["hr_lower"],
            patient_subgroups_discovery[i]["hr_upper"],
            patient_subgroups_discovery[i]["p"],
        )
    )

lengths_discovery = [
    survival_d.loc[survival_d["patientID"] == i, "time"].values[0]
    for i in patient_ids_discovery
]
statuses_discovery = [
    (survival_d.loc[survival_d["patientID"] == i, "status"].values[0])
    for i in patient_ids_discovery
]
kmf = KaplanMeierFitter()
f, ax = plt.subplots(figsize=(5, 5))
for subgroup in patient_subgroups_discovery:
    subgroup_id = subgroup["subgroup_id"]
    length_A, event_observed_A = (
        np.array(lengths_discovery)[subgroup_ids_discovery == subgroup_id],
        np.array(statuses_discovery)[subgroup_ids_discovery == subgroup_id],
    )
    label = "{} (N={})".format(
        subgroup["subgroup_id"], np.sum(subgroup_ids_discovery == subgroup_id)
    )
    kmf.fit(length_A, event_observed_A, label=label)
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        color=color_palette_Bigraph[subgroup_id],
        show_censors=True,
        linewidth=2,
        censor_styles={"ms": 5, "marker": "|"},
    )
log_rank_test = multivariate_logrank_test(
    np.array(lengths_discovery)[subgroup_ids_discovery != 0],
    np.array(subgroup_ids_discovery)[subgroup_ids_discovery != 0],
    np.array(statuses_discovery)[subgroup_ids_discovery != 0],
)
p_value = log_rank_test.p_value
ax.legend(ncol=2, fontsize=13)
ax.text(
    x=0.3,
    y=0.95,
    s="p-value = {:.5f}".format(p_value),
    fontsize=14,
    transform=ax.transAxes,
)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
sns.despine()
f.savefig("Results/figS2_c_2.png", dpi=600, bbox_inches="tight")
f.savefig("Results/figS2_c_2.svg", dpi=600, bbox_inches="tight")