import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import reverse_dict, preprocess_Danenberg, preprocess_Jackson
from definitions import (
    get_node_id,
    Protein_list_Danenberg,
    Protein_list_Jackson,
    Protein_list_display_Danenberg,
    Protein_list_display_Jackson,
    Cell_types_displayed_Danenberg,
    Cell_types_displayed_Jackson,
)
sys.path.append("./..")


SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
SC_ev_raw = pd.read_csv("Datasets/Jackson_et_al/cells.csv")
survival_ev_raw = pd.read_csv("Datasets/Jackson_et_al/clinical.csv")
SC_ev, survival_ev = preprocess_Jackson(SC_ev_raw, survival_ev_raw)

Cell_type_name_list = [
    reverse_dict(get_node_id("Danenberg", "CellType"))[i] for i in range(32)
]
SC_d_raw["meta_description"] = pd.Categorical(
    SC_d_raw["meta_description"], categories=Cell_type_name_list, ordered=True
)
expression_per_phenoptype = SC_d_raw.groupby("meta_description")[
    Protein_list_Danenberg
].median()
expression_per_phenoptype_scaled = (
    expression_per_phenoptype - expression_per_phenoptype.mean()
) / expression_per_phenoptype.std()


f, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(
    expression_per_phenoptype_scaled,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-2,
    vmax=2,
    cbar=True,
    linewidth=0.1,
)
ax.set_xticklabels(
    Protein_list_display_Danenberg,
    fontsize=12,
)

ax.set_yticklabels(Cell_types_displayed_Danenberg, fontsize=12, fontweight="bold")
# ax.set_yticklabels(ax.get_yticklabels(), fontsize = 12, fontweight='bold')
ax.set_xlabel("Antigens", fontsize=18, fontweight="bold")
ax.set_ylabel("Cell Phenotypes", fontsize=18, fontweight="bold")
ytickcolors = ["cornflowerblue"] * 16 + ["darkorange"] * 11 + ["forestgreen"] * 5
for ytick, color in zip(ax.get_yticklabels(), ytickcolors):
    ytick.set_color(color)
ax.set_title(
    "z-scored median expression of antigens per cell phenotype",
    fontsize=20,
    fontweight="bold",
)
f.savefig("Results/figA1_a.png", dpi=300, bbox_inches="tight")
f.savefig("Results/figA1_a.svg", dpi=600, bbox_inches="tight")


SC_ev_raw["cell_type"] = pd.Categorical(
    SC_ev_raw["cell_type"], categories=Cell_types_displayed_Jackson, ordered=True
)
expression_per_phenoptype = SC_ev_raw.groupby("cell_type")[
    Protein_list_Jackson
].median()
expression_per_phenoptype_scaled = (
    expression_per_phenoptype - expression_per_phenoptype.mean()
) / expression_per_phenoptype.std()


f, ax = plt.subplots(figsize=(15, 7))
sns.heatmap(
    expression_per_phenoptype_scaled,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-2,
    vmax=2,
    cbar=True,
    linewidth=0.1,
)
ax.set_xticklabels(Protein_list_display_Jackson, fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.set_xlabel("Antigens", fontsize=18, fontweight="bold")
ax.set_ylabel("Cell Phenotypes", fontsize=18, fontweight="bold")
ax.set_title(
    "z-scored median expression of antigens per cell phenotype",
    fontsize=20,
    fontweight="bold",
)
ytickcolors = ["cornflowerblue"] * 13 + ["darkorange"] * 6 + ["forestgreen"] * 7
for ytick, color in zip(ax.get_yticklabels(), ytickcolors):
    ytick.set_color(color)
f.show()
f.savefig("Results/figA1_b.png", dpi=300, bbox_inches="tight")
f.savefig("Results/figA1_b.svg", dpi=300, bbox_inches="tight")