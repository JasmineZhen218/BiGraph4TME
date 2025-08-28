import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_Wang
from definitions import (
    Protein_list_Wang,
    Cell_types_Wang
)



SC_ev_tnbc_raw = pd.read_csv("Datasets/Wang_TNBC/cells.csv")
survival_ev_tnbc_raw = pd.read_csv("Datasets/Wang_TNBC/clinical.csv")
SC_ev_tnbc, survival_ev_tnbc = preprocess_Wang(SC_ev_tnbc_raw, survival_ev_tnbc_raw)

SC_ev_tnbc_raw["cell_type"] = pd.Categorical(
    SC_ev_tnbc_raw["Label"], categories=Cell_types_Wang, ordered=True
)
expression_per_phenoptype = SC_ev_tnbc_raw.groupby("cell_type")[
    Protein_list_Wang
].median()
expression_per_phenoptype_scaled = (
    expression_per_phenoptype - expression_per_phenoptype.mean()
) / expression_per_phenoptype.std()


f, ax = plt.subplots(figsize=(15, 9))
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
ax.set_xticklabels(Protein_list_Wang, fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.set_xlabel("Antigens", fontsize=18, fontweight="bold")
ax.set_ylabel("Cell Phenotypes", fontsize=18, fontweight="bold")
ax.set_title(
    "z-scored median expression of antigens per cell phenotype",
    fontsize=20,
    fontweight="bold",
)
ytickcolors = ["cornflowerblue"] * 17 + ["darkorange"] * 16 + ["forestgreen"] * 4
for ytick, color in zip(ax.get_yticklabels(), ytickcolors):
    ytick.set_color(color)
f.show()
f.savefig("Results/figA1_c.png", dpi=300, bbox_inches="tight")
f.savefig("Results/figA1_c.svg", dpi=300, bbox_inches="tight")
