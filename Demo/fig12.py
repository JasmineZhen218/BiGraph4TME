import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from utils import preprocess_Danenberg
sys.path.append("./..")
from bi_graph import BiGraph

# --- Setup ---
os.makedirs("Results/Fig12", exist_ok=True)

# Load data
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)

# Run BiGraph
bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(SC_d, survival_data=survival_d)
patient_ids_discovery = np.array(SC_d["patientID"].unique())

# Assign clinical subtype
clinical_subtypes_discovery = np.full(len(patient_ids_discovery), "Unknown", dtype=object)
for i, pid in enumerate(patient_ids_discovery):
    row = survival_d[survival_d["patientID"] == pid].iloc[0]
    if row["HER2 Status"] == "Positive":
        clinical_subtypes_discovery[i] = "HER2+"
    elif (row["ER Status"] == "Positive" or row["PR Status"] == "Positive") and row["HER2 Status"] == "Negative":
        clinical_subtypes_discovery[i] = "HR+/HER2-"
    elif row["ER Status"] == row["PR Status"] == "Negative" and row["HER2 Status"] == "Negative":
        clinical_subtypes_discovery[i] = "TNBC"

# Compute proportions
Histograms = np.stack(bigraph_.fitted_soft_wl_subtree.Histograms, axis=0)
Histograms = np.stack(Histograms, axis=0)
Proportions = Histograms / Histograms.sum(axis=1, keepdims=True)

# Boxplot function
def plot_box(feature_name, feature_list, compare_list, subgroup_id, suffix, subset_mask=None, xlabel=None):
    idx = int(subgroup_id.split('S')[1]) - 1
    patterns = patient_subgroups_discovery[idx]["characteristic_patterns"]
    proportion = Proportions[:, patterns].sum(axis=1)
    patients = patient_ids_discovery if subset_mask is None else patient_ids_discovery[subset_mask]
    proportions = proportion if subset_mask is None else proportion[subset_mask]

    values = [survival_d[survival_d["patientID"] == pid][feature_name].values[0] for pid in patients]
    df = pd.DataFrame({"Proportion": proportions, feature_name: values}).dropna()

    f, ax = plt.subplots(figsize=(3, 3))
    sns.boxplot(x=feature_name, y="Proportion", data=df, showfliers=False, order=feature_list, color="grey")
    annot = Annotator(ax, compare_list, data=df, x=feature_name, y="Proportion", order=feature_list)
    annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
    annot.apply_test()
    ax, _ = annot.annotate()
    ax.set_ylabel("Proportion in each patient", fontsize=12)
    ax.set_xlabel(xlabel or feature_name, fontsize=12)
    f.savefig(f"Results/Fig12/fig12_{suffix}.svg", dpi=300, bbox_inches="tight")

# Run all boxplots
plot_box("Grade", [1, 2, 3], [(1, 2), (2, 3), (1, 3)], "S1", "a")
plot_box("Grade", [1, 2, 3], [(1, 2), (2, 3), (1, 3)], "S7", "e")
plot_box("Grade", [1, 2, 3], [(1, 2), (2, 3), (1, 3)], "S4", "i", clinical_subtypes_discovery == "TNBC")

plot_box("Tumor Stage", [1, 2, 3, 4], [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], "S1", "b", xlabel="Stage")
plot_box("Tumor Stage", [1, 2, 3, 4], [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], "S7", "f", xlabel="Stage")
plot_box("Tumor Stage", [1, 2, 3], [(1, 2), (1, 3), (2, 3)], "S4", "j", clinical_subtypes_discovery == "TNBC", xlabel="Stage")

# Lymphatic metastasis plots
def plot_lymphatic(subgroup_id, suffix, mask=None):
    idx = int(subgroup_id.split('S')[1]) - 1
    patterns = patient_subgroups_discovery[idx]["characteristic_patterns"]
    proportion = Proportions[:, patterns].sum(axis=1)
    patients = patient_ids_discovery if mask is None else patient_ids_discovery[mask]
    proportions = proportion if mask is None else proportion[mask]

    values = ["Yes" if survival_d[survival_d["patientID"] == pid]["Lymph nodes examined positive"].values[0] > 0 else "No" for pid in patients]
    df = pd.DataFrame({"Proportion": proportions, "Lymphatic metastasis": values}).dropna()

    f, ax = plt.subplots(figsize=(3, 3))
    sns.boxplot(x="Lymphatic metastasis", y="Proportion", data=df, showfliers=False, order=["Yes", "No"], color="grey")
    annot = Annotator(ax, [("Yes", "No")], data=df, x="Lymphatic metastasis", y="Proportion", order=["Yes", "No"])
    annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
    annot.apply_test()
    ax, _ = annot.annotate()
    ax.set_ylabel("Proportion in each patient", fontsize=12)
    ax.set_xlabel("Lymphatic metastasis", fontsize=12)
    f.savefig(f"Results/Fig12/fig12_{suffix}.svg", dpi=300, bbox_inches="tight")

plot_lymphatic("S1", "c")
plot_lymphatic("S7", "g")
plot_lymphatic("S4", "k", clinical_subtypes_discovery == "TNBC")

# Clinical subtype plots
def plot_clinical_subtype(subgroup_id, suffix):
    idx = int(subgroup_id.split('S')[1]) - 1
    patterns = patient_subgroups_discovery[idx]["characteristic_patterns"]
    proportion = Proportions[:, patterns].sum(axis=1)
    df = pd.DataFrame({"Proportion": proportion, "Clinical Subtype": clinical_subtypes_discovery}).dropna()

    f, ax = plt.subplots(figsize=(3, 3))
    order = ["HER2+", "HR+/HER2-", "TNBC"]
    compare = [("HER2+", "HR+/HER2-"), ("HER2+", "TNBC"), ("HR+/HER2-", "TNBC")]
    sns.boxplot(x="Clinical Subtype", y="Proportion", data=df, showfliers=False, order=order, color="grey")
    annot = Annotator(ax, compare, data=df, x="Clinical Subtype", y="Proportion", order=order)
    annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
    annot.apply_test()
    ax, _ = annot.annotate()
    ax.set_ylabel("Proportion in each patient", fontsize=12)
    ax.set_xlabel("Clinical Subtype", fontsize=12)
    f.savefig(f"Results/Fig12/fig12_{suffix}.svg", dpi=300, bbox_inches="tight")

plot_clinical_subtype("S1", "d")
plot_clinical_subtype("S7", "h")