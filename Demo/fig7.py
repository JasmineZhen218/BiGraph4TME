import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
from scipy.stats import spearmanr
from utils import preprocess_Danenberg
from definitions import color_palette_clinical
sys.path.append("./..")
from bi_graph import BiGraph
from population_graph import Population_Graph


SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
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
clinical_subtypes_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
clinical_subtypes_discovery[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    er = survival_d.loc[survival_d["patientID"] == patient_id, "ER Status"].values[0]
    pr = survival_d.loc[survival_d["patientID"] == patient_id, "PR Status"].values[0]
    her2 = survival_d.loc[survival_d["patientID"] == patient_id, "HER2 Status"].values[
        0
    ]
    if her2 == "Positive":
        clinical_subtypes_discovery[i] = "HER2+"  # Her2+
    if (er == "Positive" or pr == "Positive") and her2 == "Negative":
        clinical_subtypes_discovery[i] = "HR+/HER2-"  # HR+/HER2-
    elif (er == "Negative" and pr == "Negative") and her2 == "Negative":
        clinical_subtypes_discovery[i] = "TNBC"  # TNBC
print(
    "{} patients in total, {} Unkonw, {} HER2+, {} HR+/HER2-, {} TNBC".format(
        len(clinical_subtypes_discovery),
        np.sum(clinical_subtypes_discovery == "Unknown"),
        np.sum(clinical_subtypes_discovery == "HER2+"),
        np.sum(clinical_subtypes_discovery == "HR+/HER2-"),
        np.sum(clinical_subtypes_discovery == "TNBC"),
    )
)

stages_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
stages_discovery[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    stage = survival_d.loc[survival_d["patientID"] == patient_id, "Tumor Stage"].values[
        0
    ]
    if stage == 1.0:
        stages_discovery[i] = "Stage I"
    elif stage == 2.0:
        stages_discovery[i] = "Stage II"
    elif stage == 3.0:
        stages_discovery[i] = "Stage III"
    elif stage == 4.0:
        stages_discovery[i] = "Stage IV"
print(
    "{} patients in total, {} Unkonw, {} Stage I, {} Stage II, {} Stage III, {} Stage IV".format(
        len(stages_discovery),
        np.sum(stages_discovery == "Unknown"),
        np.sum(stages_discovery == "Stage I"),
        np.sum(stages_discovery == "Stage II"),
        np.sum(stages_discovery == "Stage III"),
        np.sum(stages_discovery == "Stage IV"),
    )
)

grades_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
grades_discovery[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    grade = survival_d.loc[survival_d["patientID"] == patient_id, "Grade"].values[0]
    if grade == 1.0:
        grades_discovery[i] = "Grade 1"
    elif grade == 2.0:
        grades_discovery[i] = "Grade 2"
    elif grade == 3.0:
        grades_discovery[i] = "Grade 3"

print(
    "{} patients in total, {} Unkonw, {} Grade 1, {} Grade 2, {} Grade 3".format(
        len(grades_discovery),
        np.sum(grades_discovery == "Unknown"),
        np.sum(grades_discovery == "Grade 1"),
        np.sum(grades_discovery == "Grade 2"),
        np.sum(grades_discovery == "Grade 3"),
    )
)

positive_lymph = np.zeros(len(patient_ids_discovery), dtype=object)
positive_lymph[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    lymph = survival_d.loc[
        survival_d["patientID"] == patient_id, "LymphNodesOrdinal"
    ].values[0]
    positive_lymph[i] = lymph

print(
    "{} patients in total, {} Unkonw, {} 0, {} 1, {} 2-3, {} 4-7, {} 7+".format(
        len(positive_lymph),
        np.sum(positive_lymph == "Unknown"),
        np.sum(positive_lymph == "0"),
        np.sum(positive_lymph == "1"),
        np.sum(positive_lymph == "2-3"),
        np.sum(positive_lymph == "4-7"),
        np.sum(positive_lymph == "7+"),
    )
)

ages_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
ages_discovery[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    age = survival_d.loc[
        survival_d["patientID"] == patient_id, "Age at Diagnosis"
    ].values[0]
    if age < 50:
        ages_discovery[i] = "<50"
    elif age >= 50 and age <= 70:
        ages_discovery[i] = "50-70"
    elif age > 70:
        ages_discovery[i] = ">70"
print(
    "{} patients in total, {} Unkonw, {} <50, {} 50-70, {} >70".format(
        len(ages_discovery),
        np.sum(ages_discovery == "Unknown"),
        np.sum(ages_discovery == "<50"),
        np.sum(ages_discovery == "50-70"),
        np.sum(ages_discovery == ">70"),
    )
)

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
    c=[color_palette_clinical[i] for i in clinical_subtypes_discovery],
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
for clinical_subtype_id in ["Unknown", "HER2+", "HR+/HER2-", "TNBC"]:
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markerfacecolor=color_palette_clinical[clinical_subtype_id],
            label="{} (N = {})".format(
                clinical_subtype_id,
                np.sum(clinical_subtypes_discovery == clinical_subtype_id),
            ),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )

ax.legend(handles=handles, fontsize=11, ncols=2)
ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
fig.savefig("Results/fig7_a.png", dpi=300, bbox_inches="tight")

color_palette_stage = {
    "Unknown": "grey",
    "Stage I": sns.color_palette("YlGn")[1],
    "Stage II": sns.color_palette("Set3")[4],
    "Stage III": sns.color_palette("Spectral")[2],
    "Stage IV": sns.color_palette("Set3")[3],
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
    c=[color_palette_stage[i] for i in stages_discovery],
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
for stage_id in ["Unknown", "Stage I", "Stage II", "Stage III", "Stage IV"]:
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markerfacecolor=color_palette_stage[stage_id],
            label="{} (N = {})".format(stage_id, np.sum(stages_discovery == stage_id)),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )

ax.legend(handles=handles, fontsize=11, ncols=2)
ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
fig.savefig("Results/fig7_b.png", dpi=300, bbox_inches="tight")


# For visualization purpose, we make nodes distant from each other
color_palette_grade = {
    "Unknown": "grey",
    "Grade 1": sns.color_palette("Set3")[0],
    "Grade 2": sns.color_palette("Set3")[1],
    "Grade 3": sns.color_palette("Set3")[2],
}
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
    c=[color_palette_grade[i] for i in grades_discovery],
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
for grade_id in ["Unknown", "Grade 1", "Grade 2", "Grade 3"]:
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markerfacecolor=color_palette_grade[grade_id],
            label="{} (N = {})".format(grade_id, np.sum(grades_discovery == grade_id)),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )

ax.legend(handles=handles, fontsize=11, ncols=2)
ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
fig.savefig("Results/fig7_c.png", dpi=300, bbox_inches="tight")


# For visualization purpose, we make nodes distant from each other
color_palette_age = {
    "Unknown": "grey",
    "<50": sns.color_palette("Set3")[-3],
    "50-70": sns.color_palette("Set3")[-2],
    ">70": sns.color_palette("Set3")[-1],
}
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
    c=[color_palette_age[i] for i in ages_discovery],
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
for age_id in ["Unknown", "<50", "50-70", ">70"]:
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markerfacecolor=color_palette_age[age_id],
            label="{} (N = {})".format(age_id, np.sum(ages_discovery == age_id)),
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
    )

ax.legend(handles=handles, fontsize=11, ncols=2)
ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
fig.savefig("Results/fig7_d.png", dpi=300, bbox_inches="tight")



import numpy as np

spearmanr_corr = np.zeros((13, 7))
P_value = np.zeros((13, 7))
for i in range(3):
    clinical_subtype = ["HER2+", "TNBC", "HR+/HER2-"][i]
    for j in range(7):
        subgroup_id = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"][j]
        variable_1 = np.array(clinical_subtypes_discovery) == clinical_subtype
        variable_2 = np.array(subgroup_ids_discovery) == subgroup_id
        correlation_coefficient, p_value = spearmanr(variable_1, variable_2)
        spearmanr_corr[i, j] = correlation_coefficient
        P_value[i, j] = p_value
        if p_value < 0.05 / (3 * 7):
            print(
                "Clinical Subtype: {}, Subgroup: {}, correlation_coefficient: {:.2f}, p_val: {:.2e}".format(
                    clinical_subtype, subgroup_id, correlation_coefficient, p_value
                )
            )
for i in range(3, 7):
    stage = ["Stage I", "Stage II", "Stage III", "Stage IV"][i - 3]
    for j in range(7):
        subgroup_id = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"][j]
        variable_1 = np.array(stages_discovery) == stage
        variable_2 = np.array(subgroup_ids_discovery) == subgroup_id
        correlation_coefficient, p_value = spearmanr(variable_1, variable_2)
        spearmanr_corr[i, j] = correlation_coefficient
        P_value[i, j] = p_value
        if p_value < 0.05 / (4 * 7):
            print(
                "Stage: {}, Subgroup: {}, correlation_coefficient: {:.2f}, p_val: {:.2e}".format(
                    stage, subgroup_id, correlation_coefficient, p_value
                )
            )
for i in range(7, 10):
    grade = ["Grade 1", "Grade 2", "Grade 3"][i - 7]
    for j in range(7):
        subgroup_id = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"][j]
        variable_1 = np.array(grades_discovery) == grade
        variable_2 = np.array(subgroup_ids_discovery) == subgroup_id
        correlation_coefficient, p_value = spearmanr(variable_1, variable_2)
        spearmanr_corr[i, j] = correlation_coefficient
        P_value[i, j] = p_value
        if p_value < 0.05 / (3 * 7):
            print(
                "Grade: {}, Subgroup: {}, correlation_coefficient: {:.2f}, p_val: {:.2e}".format(
                    grade, subgroup_id, correlation_coefficient, p_value
                )
            )

for i in range(10, 13):
    age = ["<50", "50-70", ">70"][i - 10]
    for j in range(7):
        subgroup_id = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"][j]
        variable_1 = np.array(ages_discovery) == age
        variable_2 = np.array(subgroup_ids_discovery) == subgroup_id
        correlation_coefficient, p_value = spearmanr(variable_1, variable_2)
        spearmanr_corr[i, j] = correlation_coefficient
        P_value[i, j] = p_value
        if p_value < 0.05 / (3 * 7):
            print(
                "Age: {}, Subgroup: {}, correlation_coefficient: {:.2f}, p_val: {:.2e}".format(
                    grade, subgroup_id, correlation_coefficient, p_value
                )
            )


f, ax = plt.subplots(figsize=(5, 8))
sns.heatmap(
    spearmanr_corr,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=False,
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    vmax=0.6,
)
ax.text(
    0 + 0.5,
    0 + 0.8,
    "****",
    horizontalalignment="center",
    verticalalignment="center",
    color="white",
)
ax.text(
    0 + 0.5,
    2 + 0.8,
    "****",
    horizontalalignment="center",
    verticalalignment="center",
    color="black",
)

ax.text(
    1 + 0.5,
    1 + 0.8,
    "****",
    horizontalalignment="center",
    verticalalignment="center",
    color="white",
)
ax.text(
    3 + 0.5,
    1 + 0.8,
    "****",
    horizontalalignment="center",
    verticalalignment="center",
    color="white",
)

ax.text(
    6 + 0.5,
    2 + 0.8,
    "***",
    horizontalalignment="center",
    verticalalignment="center",
    color="white",
)

ax.text(
    2 + 0.5,
    6 + 0.8,
    "***",
    horizontalalignment="center",
    verticalalignment="center",
    color="white",
)

ax.text(
    6 + 0.5,
    7 + 0.8,
    "***",
    horizontalalignment="center",
    verticalalignment="center",
    color="white",
)

ax.text(
    6 + 0.5,
    8 + 0.8,
    "***",
    horizontalalignment="center",
    verticalalignment="center",
    color="white",
)

ax.text(
    6 + 0.5,
    9 + 0.8,
    "***",
    horizontalalignment="center",
    verticalalignment="center",
    color="k",
)


ax.set_xlabel("BiGraph Derived Patient Subgroup")
# ax.set_ylabel("Clinical Subtype")
ax.set_xticklabels(["S1", "S2", "S3", "S4", "S5", "S6", "S7"], rotation=0)
ax.set_yticklabels(
    [
        "HER2+",
        "TNBC",
        "HR+/HER2-",
        "Satge I",
        "Satge II",
        "Satge III",
        "Satge IV",
        "Grade 1",
        "Grade 2",
        "Grade 3",
        "<50",
        "50-70",
        ">70",
    ],
    rotation=0,
)
ax.set(title="Spearman's correlation score")
plt.show()
f.savefig("Results/fig7_e.png", dpi=300, bbox_inches="tight")