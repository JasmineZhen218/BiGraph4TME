
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_Danenberg, preprocess_Wang
from definitions import get_paired_markers
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
from statannotations.Annotator import Annotator
sys.path.append("./..")
from bi_graph import BiGraph

SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)
SC_ev_tnbc_raw = pd.read_csv("Datasets/Wang_TNBC/cells.csv")
survival_ev_tnbc_raw = pd.read_csv("Datasets/Wang_TNBC/clinical.csv")
SC_ev_tnbc, survival_ev_tnbc = preprocess_Wang(SC_ev_tnbc_raw, survival_ev_tnbc_raw)

def map_cell_types(SC_d, SC_ev, paired_proteins):
    cell_types_d = SC_d["celltypeID"].values
    protein_expression_d = SC_d[[i[0] for i in paired_proteins]].values
    # normalize protein expression
    protein_expression_d = (
        protein_expression_d - np.mean(protein_expression_d, axis=0)
    ) / np.std(protein_expression_d, axis=0)
    protein_expression_ev = SC_ev[[i[1] for i in paired_proteins]].values
    # normalize protein expression
    protein_expression_ev = (
        protein_expression_ev - np.mean(protein_expression_ev, axis=0)
    ) / np.std(protein_expression_ev, axis=0)
    centroids = np.zeros((len(np.unique(cell_types_d)), len(paired_proteins)))
    for i in range(len(np.unique(cell_types_d))):
        centroids[i] = np.mean(protein_expression_d[cell_types_d == i], axis=0)
    cell_types_ev_hat = np.zeros(len(protein_expression_ev), dtype=int)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(centroids)
    # to avoid memory error, we split the data into chunks
    n_chunks = 100
    chunk_size = len(protein_expression_ev) // n_chunks
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        _, indices = neigh.kneighbors(protein_expression_ev[start:end])
        cell_types_ev_hat[start:end] = indices.flatten()
    # the last chunk
    _, indices = neigh.kneighbors(protein_expression_ev[end:])
    cell_types_ev_hat[end:] = indices.flatten()
    SC_ev["celltypeID_original"] = SC_ev["celltypeID"]
    SC_ev["celltypeID"] = list(cell_types_ev_hat)
    return SC_ev



SC_ev_tnbc_baseline = SC_ev_tnbc[SC_ev_tnbc['BiopsyPhase'] == 'Baseline']
SC_ev_tnbc_baseline = map_cell_types(
    SC_d, SC_ev_tnbc_baseline, get_paired_markers(source="Danenberg", target="Wang")
)
population_graph_ev_tnbc_baseline, patient_subgroups_ev_tnbc_baseline, histograms_ev_tnbc_baseline, Signature_ev_tnbc_baseline = bigraph_.transform(
    SC_ev_tnbc_baseline, survival_data=None
)
SC_ev_tnbc_ot = SC_ev_tnbc[SC_ev_tnbc['BiopsyPhase'] == 'On-treatment']
SC_ev_tnbc_ot = map_cell_types(
    SC_d, SC_ev_tnbc_ot, get_paired_markers(source="Danenberg", target="Wang")
)
population_graph_ev_tnbc_ot, patient_subgroups_ev_tnbc_ot, histograms_ev_tnbc_ot, Signature_ev_tnbc_ot = bigraph_.transform(
    SC_ev_tnbc_ot, survival_data=None
)


survival_ev_tnbc = survival_ev_tnbc.rename(
        columns={
            "PatientID": "patientID"}
    )
patient_ids_ev_tnbc_baseline = list(SC_ev_tnbc_baseline["patientID"].unique())
response_baseline = [survival_ev_tnbc.loc[survival_ev_tnbc["patientID"] == i, "pCR"].values[0] for i in patient_ids_ev_tnbc_baseline]
arm_baseline = [survival_ev_tnbc.loc[survival_ev_tnbc["patientID"] == i, "Arm"].values[0] for i in patient_ids_ev_tnbc_baseline]
characteristic_patterns = patient_subgroups_discovery[3][
            "characteristic_patterns"
        ]
proportions_ev_tnbc_baseline = histograms_ev_tnbc_baseline / np.sum(histograms_ev_tnbc_baseline, axis=1, keepdims=True)
df = pd.DataFrame({
    'pCR': response_baseline,
    'Proportion': proportions_ev_tnbc_baseline[:, characteristic_patterns].sum(axis = 1),
    'Existence': proportions_ev_tnbc_baseline[:, characteristic_patterns].sum(axis = 1)>=0.01,
    'Arm': arm_baseline
})
print(df.groupby(['pCR',  'Existence']).size())
a = df.groupby(['pCR', 'Existence']).size()


res = stats.mannwhitneyu(
    df.loc[(df['pCR']=='pCR') , 'Proportion'],
    df.loc[(df['pCR']=='RD'), 'Proportion'],
    alternative='greater'
)
print(res)

# show boxplot of proportion of RD and pCR
f, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.boxplot(data = df, x='pCR', y='Proportion',ax=ax, showfliers=False, palette = {
    'RD': sns.color_palette("Set2")[1],
    'pCR': sns.color_palette("Set2")[0]
})

annot = Annotator(
        ax,
        [('RD', 'pCR'),],
        data=df,
        x="pCR",
        y="Proportion",
    )
annot.configure(
        test="Mann-Whitney",
        text_format="simple",
        loc="inside",
        verbose=2,
    )
annot.apply_test()
formatted_pvalues = f'p = {res.pvalue:.5f}'
annot.set_custom_annotations([formatted_pvalues])
ax, test_results = annot.annotate()
ax.set(
    xlabel="Response",
    ylabel="Proportion of TLS niche",
    xticklabels=["RD", "pCR"],
    title = "Pre-Treatment",
)
f.savefig("Results/fig15_g.jpg", bbox_inches='tight')


patient_ids_ev_tnbc_ot = list(SC_ev_tnbc_ot["patientID"].unique())
response_ot = [survival_ev_tnbc.loc[survival_ev_tnbc["patientID"] == i, "pCR"].values[0] for i in patient_ids_ev_tnbc_ot]
arm_ot = [survival_ev_tnbc.loc[survival_ev_tnbc["patientID"] == i, "Arm"].values[0] for i in patient_ids_ev_tnbc_ot]
characteristic_patterns = patient_subgroups_discovery[3][
            "characteristic_patterns"
        ]
proportions_ev_tnbc_ot = histograms_ev_tnbc_ot / np.sum(histograms_ev_tnbc_ot, axis=1, keepdims=True)
df = pd.DataFrame({
    'pCR': response_ot,
    'Proportion': proportions_ev_tnbc_ot[:, characteristic_patterns].sum(axis = 1),
    'Existence': proportions_ev_tnbc_ot[:, characteristic_patterns].sum(axis = 1)>=0.01,
    'Arm': arm_ot
})
print(df.groupby(['pCR',  'Existence']).size())
a = df.groupby(['pCR', 'Existence']).size()
res = stats.mannwhitneyu(
    df.loc[(df['pCR']=='pCR') , 'Proportion'],
    df.loc[(df['pCR']=='RD'), 'Proportion'],
    alternative='greater'
)
print(res)

# show boxplot of proportion of RD and pCR
f, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.boxplot(data = df, x='pCR', y='Proportion',ax=ax, showfliers=False, palette = {
    'RD': sns.color_palette("Set2")[1],
    'pCR': sns.color_palette("Set2")[0]
})
annot = Annotator(
        ax,
        [('RD', 'pCR'),],
        data=df,
        x="pCR",
        y="Proportion",


    )
annot.configure(
        test="Mann-Whitney",
        text_format="simple",
        loc="inside",
        verbose=2,
        # alternative = 'two-sided',

    )
annot.apply_test()
formatted_pvalues = f'p = {res.pvalue:.5f}'
annot.set_custom_annotations([formatted_pvalues])

ax, test_results = annot.annotate()

ax.set(
    xlabel="Response",
    ylabel="Proportion of TLS-like niche",
    xticklabels=["RD", "pCR"],
    title = "On-Treatment",
)

plt.show()
f.savefig("Results/fig15_h.jpg", bbox_inches='tight')



arm_baseline = [
    survival_ev_tnbc.loc[survival_ev_tnbc["patientID"] == i, "Arm"].values[0]
    for i in patient_ids_ev_tnbc_baseline
]
characteristic_patterns = patient_subgroups_discovery[3]["characteristic_patterns"]
df = pd.DataFrame(
    {
        "pCR": response_baseline,
        "Proportion": proportions_ev_tnbc_baseline[:, characteristic_patterns].sum(
            axis=1
        ),
        "Existence": proportions_ev_tnbc_baseline[:, characteristic_patterns].sum(
            axis=1
        )
        >= 0.01,
        "Arm": arm_baseline,
    }
)
print(df.groupby(["pCR", "Existence"]).size())
a = df.groupby(["pCR", "Existence"]).size()
pvalues = []
res = stats.mannwhitneyu(
    df.loc[(df["pCR"] == "pCR") & (df["Arm"] == "C"), "Proportion"],
    df.loc[(df["pCR"] == "RD") & (df["Arm"] == "C"), "Proportion"],
    alternative="greater",
)
print(res)
pvalues.append(res.pvalue)
res = stats.mannwhitneyu(
    df.loc[(df["pCR"] == "pCR") & (df["Arm"] == "C&I"), "Proportion"],
    df.loc[(df["pCR"] == "RD") & (df["Arm"] == "C&I"), "Proportion"],
    alternative="greater",
)
print(res)
pvalues.append(res.pvalue)

# show boxplot of proportion of RD and pCR
f, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.boxplot(
    data=df,
    x="Arm",
    hue="pCR",
    y="Proportion",
    ax=ax,
    order=["C", "C&I"],
    showfliers=False,
    palette={"RD": sns.color_palette("Set2")[1], "pCR": sns.color_palette("Set2")[0]},
)
annot = Annotator(
    ax,
    [(("C", "RD"), ("C", "pCR")), (("C&I", "RD"), ("C&I", "pCR"))],
    data=df,
    x="Arm",
    hue="pCR",
    y="Proportion",
    order=["C", "C&I"],
)
annot.configure(
    test="Mann-Whitney",
    text_format="simple",
    loc="inside",
    verbose=2,
    # alternative = 'two-sided',
)
annot.apply_test()
formatted_pvalues = [f"p = {pvalue:.5f}" for pvalue in pvalues]
annot.set_custom_annotations(formatted_pvalues)
ax, test_results = annot.annotate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    title="Response",
    handles=handles,
    labels=["RD", "pCR"],
)
ax.set(
    xlabel="Therapy",
    ylabel="Proportion of TLS-like niche",
    xticklabels=["C", "C&I"],
    title="Pre-Treatment",
)
f.savefig("Results/fig15_i.jpg", bbox_inches='tight')


patient_ids_ev_tnbc_ot = list(SC_ev_tnbc_ot["patientID"].unique())
response_ot = [
    survival_ev_tnbc.loc[survival_ev_tnbc["patientID"] == i, "pCR"].values[0]
    for i in patient_ids_ev_tnbc_ot
]
arm_ot = [
    survival_ev_tnbc.loc[survival_ev_tnbc["patientID"] == i, "Arm"].values[0]
    for i in patient_ids_ev_tnbc_ot
]
characteristic_patterns = patient_subgroups_discovery[3]["characteristic_patterns"]
proportions_ev_tnbc_ot = histograms_ev_tnbc_ot / np.sum(
    histograms_ev_tnbc_ot, axis=1, keepdims=True
)
df = pd.DataFrame(
    {
        "pCR": response_ot,
        "Proportion": proportions_ev_tnbc_ot[:, characteristic_patterns[0]],
        "Existence": proportions_ev_tnbc_ot[:, characteristic_patterns[0]] >= 0.01,
        "Arm": arm_ot,
    }
)
print(df.groupby(["pCR", "Existence"]).size())
a = df.groupby(["pCR", "Existence"]).size()
pvalues = []
res = stats.mannwhitneyu(
    df.loc[(df["pCR"] == "pCR") & (df["Arm"] == "C"), "Proportion"],
    df.loc[(df["pCR"] == "RD") & (df["Arm"] == "C"), "Proportion"],
    alternative="greater",
)
print(res)
pvalues.append(res.pvalue)
res = stats.mannwhitneyu(
    df.loc[(df["pCR"] == "pCR") & (df["Arm"] == "C&I"), "Proportion"],
    df.loc[(df["pCR"] == "RD") & (df["Arm"] == "C&I"), "Proportion"],
    alternative="greater",
)
print(res)
pvalues.append(res.pvalue)

# show boxplot of proportion of RD and pCR
f, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.boxplot(data=df, x="Arm", hue="pCR", y="Proportion", ax=ax, order=["C", "C&I"], showfliers=False, 
            palette = {
                'RD': sns.color_palette("Set2")[1],
                'pCR': sns.color_palette("Set2")[0]
            })
annot = Annotator(
    ax,
    [(("C", "RD"), ("C", "pCR")), (("C&I", "RD"), ("C&I", "pCR"))],
    data=df,
    x="Arm",
    hue="pCR",
    y="Proportion",
    order=["C", "C&I"],
)
annot.configure(
    test="Mann-Whitney",
    text_format="simple",
    loc="inside",
    verbose=2,
    # alternative = 'two-sided',
)
annot.apply_test()
formatted_pvalues = [f"p = {pvalue:.5f}" for pvalue in pvalues]
annot.set_custom_annotations(formatted_pvalues)

ax, test_results = annot.annotate()

handles, labels = ax.get_legend_handles_labels()
# put the legend out of the plot
ax.legend(
    title="Outcome",
    handles=handles,
    labels=["RD", "pCR"],
    loc="upper left",
#     bbox_to_anchor=(1, 1),
)
ax.set(
    xlabel="Therapy",
    ylabel="Proportion of TLS-like niche",
    xticklabels=["C", "C&I"],
    title="On-Treatment",
)
f.savefig("Results/fig15_j.jpg", bbox_inches='tight')
