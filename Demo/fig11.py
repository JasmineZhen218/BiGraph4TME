import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from utils import preprocess_Danenberg
from definitions import color_palette_Bigraph
sys.path.append("./..")
from bi_graph import BiGraph

# --- Setup ---
os.makedirs("Results/Fig11", exist_ok=True)
# Load data
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
_, subgroups = bigraph_.fit_transform(SC_d, survival_data=survival_d)
patient_ids = list(SC_d["patientID"].unique())

# Compute survival data
Histograms = bigraph_.fitted_soft_wl_subtree.Histograms
Histograms = np.stack(Histograms, axis=0)
Proportions = Histograms / Histograms.sum(axis=1, keepdims=True)
lengths = np.array([survival_d[survival_d["patientID"] == i]["time"].values[0] for i in patient_ids])
statuses = np.array([survival_d[survival_d["patientID"] == i]["status"].values[0] for i in patient_ids])

# Plot by subgroup
threshold = 0.01
letters = list("abcdefg")
for i, subgroup in enumerate(subgroups):
    sid, patterns = subgroup["subgroup_id"], subgroup["characteristic_patterns"]
    prop = Proportions[:, patterns].sum(axis=1)

    f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
    kmf = KaplanMeierFitter()
    for label, mask, color in zip(
        ["Negative", "Positive"],
        [prop < threshold, prop >= threshold],
        ["grey", color_palette_Bigraph[sid]]
    ):
        kmf.fit(lengths[mask], statuses[mask], label=f"{label} (N = {mask.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, censor_styles={"ms": 6, "marker": "|"}, color=color, linewidth=3)

    p = logrank_test(lengths[prop < threshold], lengths[prop >= threshold], statuses[prop < threshold], statuses[prop >= threshold]).p_value
    ax.text(0.1, 0.3, f"p = {p:.5f}", fontsize=12, transform=ax.transAxes)
    ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
    ax.legend(fontsize=11)
    f.savefig(f"Results/Fig11/fig11_{letters[i]}.svg", dpi=300, bbox_inches="tight")

# Define clinical subtypes
subtypes = np.full(len(patient_ids), "Unknown", dtype=object)
for i, pid in enumerate(patient_ids):
    er = survival_d[survival_d["patientID"] == pid]["ER Status"].values[0]
    pr = survival_d[survival_d["patientID"] == pid]["PR Status"].values[0]
    her2 = survival_d[survival_d["patientID"] == pid]["HER2 Status"].values[0]
    if her2 == "Positive":
        subtypes[i] = "HER2+"
    elif (er == "Positive" or pr == "Positive") and her2 == "Negative":
        subtypes[i] = "HR+/HER2-"
    elif er == pr == "Negative" and her2 == "Negative":
        subtypes[i] = "TNBC"
print(f"{len(subtypes)} patients: {np.sum(subtypes == 'Unknown')} Unknown, {np.sum(subtypes == 'HER2+')} HER2+, {np.sum(subtypes == 'HR+/HER2-')} HR+/HER2-, {np.sum(subtypes == 'TNBC')} TNBC")

# Conditional plot for a specific subgroup and subtype
subtype_filter = "TNBC"
sid_target = "S4"
for subgroup in subgroups:
    if subgroup["subgroup_id"] != sid_target:
        continue
    patterns = subgroup["characteristic_patterns"]
    prop = Proportions[:, patterns].sum(axis=1)
    mask_neg = (prop < threshold) & (subtypes == subtype_filter)
    mask_pos = (prop >= threshold) & (subtypes == subtype_filter)

    f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
    kmf = KaplanMeierFitter()
    for label, mask, color in zip(["Negative", "Positive"], [mask_neg, mask_pos], ["grey", color_palette_Bigraph[sid_target]]):
        kmf.fit(lengths[mask], statuses[mask], label=f"{label} (N = {mask.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, censor_styles={"ms": 6, "marker": "|"}, color=color, linewidth=3)

    p = logrank_test(lengths[mask_neg], lengths[mask_pos], statuses[mask_neg], statuses[mask_pos]).p_value
    ax.text(0.1, 0.3, f"p = {p:.5f}", fontsize=12, transform=ax.transAxes)
    ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
    ax.legend(fontsize=11)
    plt.show()
    f.savefig("Results/Fig11/fig11_h.svg", dpi=300, bbox_inches="tight")
