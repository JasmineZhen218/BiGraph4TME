import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from utils import preprocess_Danenberg
from definitions import (
    color_palette_Bigraph
)
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
    len(patient_subgroups_discovery),
    1,
    figsize=(10, 8),
    sharex=True,
)
for i in range(len(patient_subgroups_discovery)):
    subgroup_id = patient_subgroups_discovery[i]["subgroup_id"]
    hodges_lehmanns = patient_subgroups_discovery[i]["hodges_lehmanns"]
    characteristic_patterns = patient_subgroups_discovery[i]["characteristic_patterns"]
    ax[i].bar(
        np.arange(len(hodges_lehmanns)),
        np.array(hodges_lehmanns)[tme_pattern_orders],
        color=[
            (
                color_palette_Bigraph[subgroup_id]
                if i in characteristic_patterns
                else "grey"
            )
            for i in tme_pattern_orders
        ],
    )
    handles = [
        patches.Rectangle(
            (0.6, 0.2),
            1,
            1,
            edgecolor=color_palette_Bigraph[subgroup_id],
            facecolor=color_palette_Bigraph[subgroup_id],
            label="Characteristic pattern in {}".format(subgroup_id),
        ),
    ]
    ax[i].legend(handles=handles, fontsize=8, loc="upper right")
    ax[i].set_ylabel("", fontsize=8)
    ax[i].get_xaxis().set_visible(False)
    ax[i].set(xlim=(-1, 61))
    ax[i].hlines(
        0.5 * np.max(hodges_lehmanns),
        0,
        len(hodges_lehmanns),
        color="k",
        linestyle="--",
        alpha=0.5,
    )
    ax[i].hlines(0, 0, len(hodges_lehmanns), color="k", alpha=0.5)


ax[0].set_title(
    "Hodges-Lehmann statistics of TME pattern expressions", fontweight="bold"
)
ax[-1].get_xaxis().set_visible(True)
ax[-1].set_xlabel("Pattern IDs", fontsize=10, fontweight="bold")
ax[-1].set(
    xticks=[0, 10, 20, 30, 40, 50, 60],
    xticklabels=[0, 10, 20, 30, 40, 50, 60],
)
ax[-1].set_xlabel("Pattern IDs", fontsize=12)
f.text(
    0.04,
    0.5,
    "Hodges-Lehmann Statistic",
    va="center",
    rotation="vertical",
    fontweight="bold",
    fontsize=12,
)
plt.show()
f.savefig("Results/figA4.png", dpi=300, bbox_inches="tight")
f.savefig("Results/figA4.svg", dpi=300, bbox_inches="tight")