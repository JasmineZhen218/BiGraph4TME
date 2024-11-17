import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sys.path.append("./..")
sys.path.append("./../..")
from utils import preprocess_Danenberg
from bi_graph import BiGraph
from definitions import Cell_types_displayed_Danenberg
    
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)

Signatures = bigraph_.fitted_soft_wl_subtree.Signatures
Histograms = bigraph_.fitted_soft_wl_subtree.Histograms
Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True)
# Proportion of each pattern presented in each patient's cellular graphs
DF_proportion = pd.DataFrame(Proportions)
DF_proportion = DF_proportion.melt(var_name="pattern_id", value_name="proportion")
DF_proportion["pattern_id"] = DF_proportion["pattern_id"].astype(str)
# Define tumor niches, immune niches, stromal niches, and interfacing niches
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
    3,
    figsize=(12, 10),
    tight_layout=True,
    gridspec_kw={"width_ratios": [1, 10, 0.3]},
)

sns.heatmap(
    Signatures[tme_pattern_orders, :],
    ax=ax[1],
    cmap="rocket_r",
    linewidth=0.005,
    cbar=True,
    cbar_ax=ax[2],
    edgecolor="black",
    vmax=np.percentile(Signatures, 99),
    vmin=np.percentile(Signatures, 5),
)

ax[1].get_yaxis().set_visible(False)
ax[1].set_title("Signature map", fontsize=14)
ax[1].set_xticklabels(
    Cell_types_displayed_Danenberg, rotation=90, fontsize=12, fontweight="bold"
)
ax[1].set_xlabel("Cell Phenotypes", fontsize=14)
xtickcolors = ["cornflowerblue"] * 16 + ["darkorange"] * 11 + ["seagreen"] * 5
for xtick, color in zip(ax[1].get_xticklabels(), xtickcolors):
    xtick.set_color(color)
ax[2].get_xaxis().set_visible(False)

sns.boxplot(
    y="pattern_id",
    x="proportion",
    data=DF_proportion,
    showfliers=False,
    ax=ax[0],
    color="white",
    order=[str(i) for i in tme_pattern_orders],
    fliersize=0.5,
)
ax[0].set_xlabel("Proportion", fontsize=12)
ax[0].set_title("Abundance", fontsize=14)
ax[0].set_yticklabels([i + 1 for i in range(len(tme_pattern_orders))], fontsize=9)
ax[0].set_ylabel("Pattern ID", fontsize=14)
ax[0].invert_xaxis()
f.savefig("Results/fig4_a.jpg", dpi=300, bbox_inches="tight")

Signatures = bigraph_.fitted_soft_wl_subtree.Signatures
Histograms = bigraph_.fitted_soft_wl_subtree.Histograms
Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True)
# Proportion of each pattern presented in each patient's cellular graphs
DF_proportion = pd.DataFrame(Proportions)
DF_proportion = DF_proportion.melt(var_name="pattern_id", value_name="proportion")
DF_proportion["pattern_id"] = DF_proportion["pattern_id"].astype(str)
# Define tumor niches, immune niches, stromal niches, and interfacing niches
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

Histograms_category = []
for i in range(len(Histograms)):
    Histograms_category.append(
        np.array([
            np.sum(Histograms[i][tumor_niche]),
            np.sum(Histograms[i][immune_niche]),
            np.sum(Histograms[i][stromal_niche]),
            np.sum(Histograms[i][ interacting_niche]),
        ])
    )
patient_ids_discovery = list(SC_d["patientID"].unique())
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
# check the proportion of each category in each patient
DF_proportion_category = pd.DataFrame(
    {
        "Tumor niche": np.sum(Proportions[:, tumor_niche], axis=1),
        "Immune niche": np.sum(Proportions[:, immune_niche], axis=1),
        "Stromal niche": np.sum(Proportions[:, stromal_niche], axis=1),
        "Interface niche": np.sum(Proportions[:, interacting_niche], axis=1),
        "Clinical Subtype": clinical_subtypes_discovery,
    }
)
DF_proportion_category = DF_proportion_category.sort_values(
    by="Clinical Subtype", 
    key=lambda x: x.map(
        {
            "HER2+": 0,
            "HR+/HER2-": 1,
            "TNBC": 2,
            "Unknown": 3,
        }
    ),


).reset_index(drop=True)
n_Her2 = np.sum(clinical_subtypes_discovery == "HER2+")
n_HR_Her2 = np.sum(clinical_subtypes_discovery == "HR+/HER2-")
n_TNBC = np.sum(clinical_subtypes_discovery == "TNBC")
n_Unknown = np.sum(clinical_subtypes_discovery == "Unknown")
# stacked bar plot
f, ax = plt.subplots(figsize=(30, 5))

# add bounding box for Her2+, HR+/Her2-, TNBC, and Unknown

ax.bar(
    DF_proportion_category.index,
    DF_proportion_category["Tumor niche"],
    color=sns.color_palette("Set3")[0],
    label="Tumor niche",
)
ax.bar(
    DF_proportion_category.index,
    DF_proportion_category["Immune niche"],
    bottom=DF_proportion_category["Tumor niche"],
    color=sns.color_palette("Set3")[1],
    label="Immune niche",
)
ax.bar(
    DF_proportion_category.index,
    DF_proportion_category["Stromal niche"],
    bottom=DF_proportion_category["Immune niche"]
    + DF_proportion_category["Tumor niche"],
    color=sns.color_palette("Set3")[2],
    label="Stromal niche",
)
ax.bar(
    DF_proportion_category.index,
    DF_proportion_category["Interface niche"],
    bottom=DF_proportion_category["Tumor niche"]
    + DF_proportion_category["Immune niche"]
    + DF_proportion_category["Stromal niche"],
    color=sns.color_palette("Set3")[3],
    label="Interface niche",
)
ax.add_patch(
    patches.Rectangle(
        (0, 0),
        n_Her2,
        1,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
      
    )
)
ax.add_patch(
     patches.Rectangle(
        (n_Her2, 0),
        n_HR_Her2,
        1,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
      
    )
)   
ax.add_patch(
     patches.Rectangle(
        (n_Her2+ n_HR_Her2, 0),
        n_TNBC,
        1,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
    )
)   
ax.set_xticks(range(0, len(DF_proportion_category.index), 10))
ax.set(xlim=(0, len(DF_proportion_category.index)), ylim=(0, 1))
ax.set_xlabel("Patient ID", fontsize=16)
ax.set_ylabel("Proportion", fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=16, ncols = 4)
f.savefig("Results/fig4_b.jpg", dpi=300, bbox_inches="tight")