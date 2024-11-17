import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_Danenberg
from definitions import Cell_types_displayed_Danenberg
sys.path.append("./..")
from bi_graph import BiGraph

SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)

Signatures = bigraph_.fitted_soft_wl_subtree.Signatures
for i in range(len(patient_subgroups_discovery)):
    subgroup_id = patient_subgroups_discovery[i]["subgroup_id"]
    characteristic_patterns = patient_subgroups_discovery[i]["characteristic_patterns"]
    if len(characteristic_patterns) == 0:
        continue
    f, ax = plt.subplots(1, 1, figsize=(10, 0.25 * len(characteristic_patterns)))
    sns.heatmap(
        Signatures[np.array(characteristic_patterns), :],
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
        # ax.set_xlabel("Cell Phenotypes", fontsize=14)
        xtickcolors = ["cornflowerblue"] * 16 + ["darkorange"] * 11 + ["seagreen"] * 5
        for xtick, color in zip(ax.get_xticklabels(), xtickcolors):
            xtick.set_color(color)
    else:
        ax.get_xaxis().set_visible(False)
    # if subgroup_id == "S1":
    #     ax.set_title(
    #         "Signature of characteristic patterns in each patient subgroups \n\n {}".format(subgroup_id), fontsize=14
    #     )
    # else:
    ax.set_title("{}".format(subgroup_id))
    # ax.set_title(f"Signature of characteristic patterns in {subgroup_id}", fontsize=10)
    plt.show()
    f.savefig("Results/fig9_{}.png".format(subgroup_id), dpi=300, bbox_inches="tight")