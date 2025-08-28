import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
from scipy.stats import spearmanr

sys.path.extend(["./..", "./../.."])
from utils import preprocess_Danenberg
from definitions import color_palette_clinical
from bi_graph import BiGraph
from population_graph import Population_Graph

os.makedirs("Results/Fig7", exist_ok=True)

# --- Load and preprocess data ---
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, _, survival_d, _ = preprocess_Danenberg(SC_d_raw, survival_d_raw)

# --- Fit BiGraph model ---
bigraph = BiGraph(k_patient_clustering=30)
_, subgroups = bigraph.fit_transform(SC_d, survival_data=survival_d)
patient_ids = SC_d["patientID"].unique()

# --- Helper function to assign categories ---
def assign_categories(patient_ids, survival_df, column, mapping, default="Unknown"):
    assigned = np.full(len(patient_ids), default, dtype=object)
    for i, pid in enumerate(patient_ids):
        val = survival_df.loc[survival_df["patientID"] == pid, column].values[0]
        for condition, label in mapping.items():
            if condition(val):
                assigned[i] = label
                break
    return assigned

# --- Assign subgroup ids ---
subgroup_ids = np.full(len(patient_ids), "Unclassified", dtype=object)
for sg in subgroups:
    idx = np.isin(patient_ids, sg["patient_ids"])
    subgroup_ids[idx] = sg["subgroup_id"]

# --- Assign clinical subtypes ---
def subtype_mapping(er, pr, her2):
    if her2 == "Positive": return "HER2+"
    if (er == "Positive" or pr == "Positive") and her2 == "Negative": return "HR+/HER2-"
    if er == "Negative" and pr == "Negative" and her2 == "Negative": return "TNBC"
    return "Unknown"

clinical_subtypes = np.array([
    subtype_mapping(
        survival_d.loc[survival_d["patientID"] == pid, "ER Status"].values[0],
        survival_d.loc[survival_d["patientID"] == pid, "PR Status"].values[0],
        survival_d.loc[survival_d["patientID"] == pid, "HER2 Status"].values[0]
    )
    for pid in patient_ids
])

# --- Assign stage, grade, lymph node, and age groups ---
stage_map = {
    1.0: "Stage I", 2.0: "Stage II", 3.0: "Stage III", 4.0: "Stage IV"
}
grade_map = {
    1.0: "Grade 1", 2.0: "Grade 2", 3.0: "Grade 3"
}

stages = np.array([stage_map.get(survival_d.loc[survival_d["patientID"] == pid, "Tumor Stage"].values[0], "Unknown") for pid in patient_ids])
grades = np.array([grade_map.get(survival_d.loc[survival_d["patientID"] == pid, "Grade"].values[0], "Unknown") for pid in patient_ids])
lymphs = np.array([survival_d.loc[survival_d["patientID"] == pid, "LymphNodesOrdinal"].values[0] for pid in patient_ids])
ages = np.array([
    "<50" if age < 50 else "50-70" if age <= 70 else ">70"
    for age in survival_d.loc[survival_d["patientID"].isin(patient_ids), "Age at Diagnosis"].values
])

# --- Print category counts ---
def print_counts(label, values, categories):
    print(f"{len(values)} patients total in {label}:")
    for cat in categories:
        print(f"  {cat}: {np.sum(values == cat)}")

print_counts("Clinical Subtypes", clinical_subtypes, ["Unknown", "HER2+", "HR+/HER2-", "TNBC"])
print_counts("Stages", stages, ["Unknown", "Stage I", "Stage II", "Stage III", "Stage IV"])
print_counts("Grades", grades, ["Unknown", "Grade 1", "Grade 2", "Grade 3"])
print_counts("Ages", ages, ["Unknown", "<50", "50-70", ">70"])

# --- Visualization function ---
def plot_population_graph(values, color_palette, filename, label_order):
    G = Population_Graph(k_clustering=20).generate(bigraph.Similarity_matrix, patient_ids)
    pos = nx.spring_layout(G, seed=3, k=1 / np.sqrt(len(G)) * 10, iterations=100, dim=3)
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    fig = plt.figure(figsize=(5, 5), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        *node_xyz.T,
        s=60,
        c=[color_palette[v] for v in values],
        edgecolors="black",
        linewidths=1,
        alpha=1,
    )
    for i, (u, v) in enumerate(G.edges()):
        weight = G[u][v]['weight']
        if weight > 0.1:
            ax.plot(*edge_xyz[i].T, alpha=0.2 * weight, color="k")

    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.set_xlim(node_xyz[:, 0].min(), node_xyz[:, 0].max())
    ax.set_ylim(node_xyz[:, 1].min(), node_xyz[:, 1].max())
    ax.set_zlim(node_xyz[:, 2].min(), node_xyz[:, 2].max())
    
    # Add legend
    handles = [
        Line2D(
            [0], [0], marker="o", color="grey",
            markerfacecolor=color_palette[label],
            label=f"{label} (N = {np.sum(values == label)})",
            markeredgecolor="black", markeredgewidth=1, markersize=8
        ) for label in label_order
    ]
    ax.legend(handles=handles, fontsize=11, ncols=2)
    # Save graph
    fig.savefig(f"Results/Fig7/{filename}.svg", dpi=600, bbox_inches="tight")
    plt.close()

# --- Apply visualization to each clinical factor ---
plot_population_graph(clinical_subtypes, color_palette_clinical, "fig7_a", ["Unknown", "HER2+", "HR+/HER2-", "TNBC"])

color_palette_stage = {
    "Unknown": "grey", "Stage I": sns.color_palette("YlGn")[1], "Stage II": sns.color_palette("Set3")[4],
    "Stage III": sns.color_palette("Spectral")[2], "Stage IV": sns.color_palette("Set3")[3],
}
plot_population_graph(stages, color_palette_stage, "fig7_b", list(color_palette_stage))

color_palette_grade = {
    "Unknown": "grey", "Grade 1": sns.color_palette("Set3")[0],
    "Grade 2": sns.color_palette("Set3")[1], "Grade 3": sns.color_palette("Set3")[2],
}
plot_population_graph(grades, color_palette_grade, "fig7_c", list(color_palette_grade))

color_palette_age = {
    "Unknown": "grey", "<50": sns.color_palette("Set3")[-3],
    "50-70": sns.color_palette("Set3")[-2], ">70": sns.color_palette("Set3")[-1],
}
plot_population_graph(ages, color_palette_age, "fig7_d", list(color_palette_age))
