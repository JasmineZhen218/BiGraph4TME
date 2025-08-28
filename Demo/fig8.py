import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import pairwise_logrank_test
from utils import preprocess_Danenberg
from definitions import color_palette_Bigraph, color_palette_clinical

sys.path.append("./..")
from bi_graph import BiGraph

# --- Setup ---
os.makedirs("Results/Fig8", exist_ok=True)

# --- Data Preparation ---
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, _, survival_d, _ = preprocess_Danenberg(SC_d_raw, survival_d_raw)

bigraph = BiGraph(k_patient_clustering=30)
_, subgroups = bigraph.fit_transform(SC_d, survival_data=survival_d)

patient_ids = SC_d["patientID"].unique()
subgroup_ids = np.full(len(patient_ids), "Unclassified", dtype=object)
for sg in subgroups:
    idx = np.isin(patient_ids, sg["patient_ids"])
    subgroup_ids[idx] = sg["subgroup_id"]

lengths = np.array([survival_d.loc[survival_d["patientID"] == pid, "time"].values[0] for pid in patient_ids])
statuses = np.array([survival_d.loc[survival_d["patientID"] == pid, "status"].values[0] for pid in patient_ids])

def assign_subtype(er, pr, her2):
    if her2 == "Positive": return "HER2+"
    if (er == "Positive" or pr == "Positive") and her2 == "Negative": return "HR+/HER2-"
    if er == "Negative" and pr == "Negative" and her2 == "Negative": return "TNBC"
    return "Unknown"

clinical_subtypes = np.array([
    assign_subtype(
        survival_d.loc[survival_d["patientID"] == pid, "ER Status"].values[0],
        survival_d.loc[survival_d["patientID"] == pid, "PR Status"].values[0],
        survival_d.loc[survival_d["patientID"] == pid, "HER2 Status"].values[0],
    ) for pid in patient_ids
])

print("{} patients: {} Unknown, {} HER2+, {} HR+/HER2-, {} TNBC".format(
    len(clinical_subtypes),
    np.sum(clinical_subtypes == "Unknown"),
    np.sum(clinical_subtypes == "HER2+"),
    np.sum(clinical_subtypes == "HR+/HER2-"),
    np.sum(clinical_subtypes == "TNBC")))

# --- Survival Analysis Function ---
def plot_survival_by_intersection(clinical_subtype, subgroup_id, fig_id):
    f, ax = plt.subplots(figsize=(4, 4))
    kmf = KaplanMeierFitter()

    # A: intersection
    mask_a = (subgroup_ids == subgroup_id) & (clinical_subtypes == clinical_subtype)
    kmf.fit(lengths[mask_a], statuses[mask_a], label=f"{clinical_subtype} ∩ {subgroup_id} (N={mask_a.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color=color_palette_Bigraph[subgroup_id],
                               show_censors=True, censor_styles={"ms": 7, "marker": "|"}, linewidth=3)

    # B: same subtype but not subgroup
    mask_b = (subgroup_ids != subgroup_id) & (clinical_subtypes == clinical_subtype)
    kmf.fit(lengths[mask_b], statuses[mask_b], label=f"{clinical_subtype} \ {subgroup_id} (N={mask_b.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color=color_palette_clinical[clinical_subtype],
                               show_censors=True, censor_styles={"ms": 7, "marker": "|"}, linewidth=3)

    # C: all other patients
    mask_c = clinical_subtypes != clinical_subtype
    kmf.fit(lengths[mask_c], statuses[mask_c], label=f"Others (N={mask_c.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color="grey",
                               show_censors=True, censor_styles={"ms": 7, "marker": "|"}, linewidth=3)

    # Log-rank test
    group_labels = np.full_like(clinical_subtypes, 1, dtype=int)
    group_labels[mask_b] = 2
    group_labels[mask_a] = 3
    result = pairwise_logrank_test(lengths, group_labels, statuses)
    print(result.summary)

    ax.legend(fontsize=12)
    ax.set_xlabel("Time (Month)", fontsize=14)
    ax.set_ylabel("Cumulative Survival", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Intersection of {clinical_subtype} and {subgroup_id}", fontsize=14)
    f.savefig(f"Results/Fig8/fig8_{fig_id}.png", dpi=600, bbox_inches="tight")
    plt.close()

# --- Run Analysis for All Intersections ---
plot_survival_by_intersection("HER2+", "S1", "a")
plot_survival_by_intersection("TNBC", "S2", "b")
plot_survival_by_intersection("HR+/HER2-", "S7", "c")
