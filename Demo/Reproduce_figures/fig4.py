import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

sys.path.extend(["./..", "./../.."])
from utils import preprocess_Danenberg
from bi_graph import BiGraph
from definitions import Cell_types_displayed_Danenberg

# --- Setup ---
os.makedirs("Results/Fig4", exist_ok=True)
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)

# --- Run BiGraph ---
bigraph = BiGraph(k_patient_clustering = 30)
population_graph, patient_subgroups = bigraph.fit_transform(SC_d, survival_data=survival_d)

# --- Signature Postprocessing ---
Signatures = bigraph.fitted_soft_wl_subtree.Signatures
Histograms = bigraph.fitted_soft_wl_subtree.Histograms
Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True)
DF_proportion = pd.DataFrame(Proportions).melt(var_name="pattern_id", value_name="proportion")
DF_proportion["pattern_id"] = DF_proportion["pattern_id"].astype(str)

# --- Define Niche Categories ---
def define_niches(Signatures, threshold=0.5):
    tumor = np.where((np.sum(Signatures[:, :16] > threshold, axis=1) > 0) &
                     (np.sum(Signatures[:, 16:] > threshold, axis=1) == 0))[0]
    immune = np.where((np.sum(Signatures[:, :16] > threshold, axis=1) == 0) &
                      (np.sum(Signatures[:, 16:27] > threshold, axis=1) > 0) &
                      (np.sum(Signatures[:, 27:] > threshold, axis=1) == 0))[0]
    stromal = np.where((np.sum(Signatures[:, :16] > threshold, axis=1) == 0) &
                       (np.sum(Signatures[:, 16:27] > threshold, axis=1) == 0) &
                       (np.sum(Signatures[:, 27:] > threshold, axis=1) > 0))[0]
    interacting = list(set(range(Signatures.shape[0])) - set(np.concatenate([tumor, immune, stromal])))
    return tumor, immune, stromal, interacting

tumor_niche, immune_niche, stromal_niche, interacting_niche = define_niches(Signatures)
tme_pattern_orders = np.concatenate([tumor_niche, immune_niche, stromal_niche, interacting_niche])

print(f"There are {Signatures.shape[0]} identified TME patterns.")
print(f"Tumor: {len(tumor_niche)}, Immune: {len(immune_niche)}, Stromal: {len(stromal_niche)}, Interacting: {len(interacting_niche)}")

# --- Plot Heatmap & Boxplot (Fig4a) ---
fig, ax = plt.subplots(1, 3, figsize=(12, 10), tight_layout=True,
                       gridspec_kw={"width_ratios": [1, 10, 0.3]})
sns.heatmap(Signatures[tme_pattern_orders], ax=ax[1], cmap="rocket_r", linewidth=0.005,
            cbar=True, cbar_ax=ax[2], edgecolor="black",
            vmax=np.percentile(Signatures, 99), vmin=np.percentile(Signatures, 5))
ax[1].set_title("Signature map", fontsize=14)
ax[1].set_xticklabels(Cell_types_displayed_Danenberg, rotation=90, fontsize=12, fontweight="bold")
for tick, color in zip(ax[1].get_xticklabels(), ["cornflowerblue"]*16 + ["darkorange"]*11 + ["seagreen"]*5):
    tick.set_color(color)
ax[1].get_yaxis().set_visible(False)
ax[2].get_xaxis().set_visible(False)

sns.boxplot(y="pattern_id", x="proportion", data=DF_proportion, ax=ax[0],
            showfliers=False, color="white", order=[str(i) for i in tme_pattern_orders])
ax[0].invert_xaxis()
ax[0].set_xlabel("Proportion", fontsize=12)
ax[0].set_title("Abundance", fontsize=14)
ax[0].set_yticks(range(0, len(tme_pattern_orders), 5))
ax[0].set_yticklabels([i+1 for i in range(0, len(tme_pattern_orders), 5)], fontsize=12)
ax[0].set_ylabel("Pattern ID", fontsize=14)

fig.savefig("Results/Fig4/fig4_a.svg", dpi=600, bbox_inches="tight")

# --- Clinical Subtypes ---
def assign_clinical_subtypes(survival_df, patient_ids):
    subtypes = np.full(len(patient_ids), "Unknown", dtype=object)
    for i, pid in enumerate(patient_ids):
        er, pr, her2 = survival_df.loc[survival_df["patientID"] == pid, ["ER Status", "PR Status", "HER2 Status"]].values[0]
        if her2 == "Positive":
            subtypes[i] = "HER2+"
        elif (er == "Positive" or pr == "Positive"):
            subtypes[i] = "HR+/HER2-"
        elif (er == "Negative" and pr == "Negative"):
            subtypes[i] = "TNBC"
    return subtypes

clinical_subtypes = assign_clinical_subtypes(survival_d, SC_d["patientID"].unique())
DF_prop_cat = pd.DataFrame({
    "Tumor niche": np.sum(Proportions[:, tumor_niche], axis=1),
    "Immune niche": np.sum(Proportions[:, immune_niche], axis=1),
    "Stromal niche": np.sum(Proportions[:, stromal_niche], axis=1),
    "Interface niche": np.sum(Proportions[:, interacting_niche], axis=1),
    "Clinical Subtype": clinical_subtypes
})
order_map = {"HER2+": 0, "HR+/HER2-": 1, "TNBC": 2, "Unknown": 3}
DF_prop_cat = DF_prop_cat.sort_values(by="Clinical Subtype", key=lambda x: x.map(order_map)).reset_index(drop=True)

# --- Stacked Bar Plot (Fig4b) ---
fig, ax = plt.subplots(figsize=(20, 5))
bottoms = np.zeros(len(DF_prop_cat))
colors = sns.color_palette("Set3")[:4]
for col, color in zip(["Tumor niche", "Immune niche", "Stromal niche", "Interface niche"], colors):
    ax.bar(DF_prop_cat.index, DF_prop_cat[col], bottom=bottoms, label=col, color=color)
    bottoms += DF_prop_cat[col].values

# Clinical subtype box annotations
subtype_counts = [np.sum(clinical_subtypes == s) for s in ["HER2+", "HR+/HER2-", "TNBC", "Unknown"]]
offset = 0
for count in subtype_counts:
    ax.add_patch(patches.Rectangle((offset, 0), count, 1, edgecolor="black", facecolor="none", linewidth=2))
    offset += count

# Aesthetics
ax.set(xlim=(0, len(DF_prop_cat)), ylim=(0, 1))
ax.set_xlabel("Patient ID", fontsize=16)
ax.set_ylabel("Proportion (%)", fontsize=16)
ax.set_xticks(range(0, len(DF_prop_cat), 32))
ax.set_xticklabels([i + 1 for i in range(0, len(DF_prop_cat), 32)], fontsize=16)
ax.set_yticks(np.arange(0, 1, 0.2))
ax.set_yticklabels([f"{i*100:.1f}" for i in np.arange(0, 1, 0.2)], fontsize=16)
plt.legend(loc="lower left", bbox_to_anchor=(0.1, 1), fontsize=16, ncols=4)

fig.savefig("Results/Fig4/fig4_b.svg", dpi=600, bbox_inches="tight")
