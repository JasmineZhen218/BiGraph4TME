import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_Danenberg
from definitions import Cell_types_displayed_Danenberg
sys.path.append("./..")
from bi_graph import BiGraph

# --- Setup ---
os.makedirs("Results/Fig9", exist_ok=True)

# Load and preprocess data
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)

# Run BiGraph model
bigraph_ = BiGraph(k_patient_clustering=30)
_, patient_subgroups_discovery = bigraph_.fit_transform(SC_d, survival_data=survival_d)
Signatures = bigraph_.fitted_soft_wl_subtree.Signatures

# Plot signatures for each subgroup
def plot_signature(subgroup_id, patterns, signature_matrix):
    if not patterns:
        return

    fig, ax = plt.subplots(figsize=(10, 0.25 * len(patterns)))
    sns.heatmap(
        signature_matrix[np.array(patterns), :],
        ax=ax,
        cbar=False,
        cmap="rocket_r",
        linewidth=0.005,
        vmax=np.percentile(Signatures, 99),
        vmin=np.percentile(Signatures, 1),
    )
    ax.get_yaxis().set_visible(False)

    if subgroup_id == "S7":
        ax.set_xticklabels(
            Cell_types_displayed_Danenberg, rotation=90, fontsize=12, fontweight="bold"
        )
        xtickcolors = ["cornflowerblue"] * 16 + ["darkorange"] * 11 + ["seagreen"] * 5
        for xtick, color in zip(ax.get_xticklabels(), xtickcolors):
            xtick.set_color(color)
    else:
        ax.get_xaxis().set_visible(False)

    ax.set_title(f"{subgroup_id}")
    fig.savefig(f"Results/Fig9/fig9_{subgroup_id}.svg", dpi=300, bbox_inches="tight")
    plt.show()

# Generate all plots
for subgroup in patient_subgroups_discovery:
    plot_signature(subgroup["subgroup_id"], subgroup["characteristic_patterns"], Signatures)