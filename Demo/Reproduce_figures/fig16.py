import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from utils import preprocess_Danenberg, preprocess_Jackson
from definitions import (
    get_paired_markers,
)
sys.path.append("./..")
from bi_graph import BiGraph

SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
SC_ev_raw = pd.read_csv("Datasets/Jackson_et_al/cells.csv")
survival_ev_raw = pd.read_csv("Datasets/Jackson_et_al/clinical.csv")
SC_ev, survival_ev = preprocess_Jackson(SC_ev_raw, survival_ev_raw)
bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)
population_graph_iv, patient_subgroups_iv, histograms_iv, Signature_iv = bigraph_.transform(
    SC_iv, survival_data=survival_iv
)
from sklearn.neighbors import NearestNeighbors
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
SC_ev = map_cell_types(
    SC_d, SC_ev, get_paired_markers(source="Danenberg", target="Jackson")
)
population_graph_ev, patient_subgroups_ev, histograms_ev, Signature_ev = bigraph_.transform(
    SC_ev, survival_data=survival_ev
)

feature_name = "Grade"
feature_list = [1, 2, 3]
compare_list = [(1, 2), (2, 3), (1, 3)]

Histograms = histograms_iv
Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True)
patient_ids_iv = list(SC_iv["patientID"].unique())

subgroup_id = 'S1'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: [
                    survival_iv.loc[
                        survival_iv["patientID"] == patient_id, feature_name
                    ].values[0]
                    for patient_id in patient_ids_iv
                ],
            }
        )
DF_presentation = DF_presentation.dropna()
f, ax = plt.subplots(figsize=(3, 3))
sns.boxplot(
        x=feature_name,
        y="Proportion",
        data=DF_presentation,
        showfliers=False,
        order=feature_list,
        color = 'grey',
    )

annot = Annotator(
        ax,
        compare_list,
        data=DF_presentation,
        x=feature_name,
        y="Proportion",
        order=feature_list,
    )
annot.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        verbose=2,
    )
annot.apply_test()
ax, test_results = annot.annotate()
ax.set_ylabel("Proportion in each patient", fontsize=12)
ax.set_xlabel("Grade", fontsize=12)
f.savefig(
    "Results/fig16_a.jpg",
    dpi=300,
    bbox_inches="tight",
)
f.savefig(
    "Results/fig16_a.svg",
    dpi=300,
    bbox_inches="tight",
)


subgroup_id = 'S7'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: [
                    survival_iv.loc[
                        survival_iv["patientID"] == patient_id, feature_name
                    ].values[0]
                    for patient_id in patient_ids_iv
                ],
            }
        )
DF_presentation = DF_presentation.dropna()
f, ax = plt.subplots(figsize=(3, 3))
sns.boxplot(
        x=feature_name,
        y="Proportion",
        data=DF_presentation,
        showfliers=False,
        order=feature_list,
        color = 'grey',
    )
annot = Annotator(
        ax,
        compare_list,
        data=DF_presentation,
        x=feature_name,
        y="Proportion",
        order=feature_list,
    )
annot.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        verbose=2,
    )
annot.apply_test()
ax, test_results = annot.annotate()
ax.set_ylabel("Proportion in each patient", fontsize=12)
ax.set_xlabel("Grade", fontsize=12)
f.savefig(
    "Results/fig16_b.jpg",
    dpi=300,
    bbox_inches="tight",
)
f.savefig(
    "Results/fig16_b.svg",
    dpi=300,
    bbox_inches="tight",
)



feature_name = "grade"
feature_list = [1, 2, 3]
compare_list = [(1, 2), (2, 3), (1, 3)]
Histograms = histograms_ev
Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True)
patient_ids_ev = list(SC_ev["patientID"].unique())
subgroup_id = 'S1'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: [
                    survival_ev.loc[
                        survival_ev["patientID"] == patient_id, feature_name
                    ].values[0]
                    for patient_id in patient_ids_ev
                ],
            }
        )
DF_presentation = DF_presentation.dropna()
f, ax = plt.subplots(figsize=(3, 3))
sns.boxplot(
        x=feature_name,
        y="Proportion",
        data=DF_presentation,
        showfliers=False,
        order=feature_list,
        color = 'grey',
    )

annot = Annotator(
        ax,
        compare_list,
        data=DF_presentation,
        x=feature_name,
        y="Proportion",
        order=feature_list,
    )
annot.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        verbose=2,
    )
annot.apply_test()
ax, test_results = annot.annotate()
ax.set_ylabel("Proportion in each patient", fontsize=12)
ax.set_xlabel("Grade", fontsize=12)
f.savefig(
    "Results/fig16_c.jpg",
    dpi=300,
    bbox_inches="tight",
)
f.savefig(
    "Results/fig16_c.svg",
    dpi=300,
    bbox_inches="tight",
)


subgroup_id = 'S7'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: [
                    survival_ev.loc[
                        survival_ev["patientID"] == patient_id, feature_name
                    ].values[0]
                    for patient_id in patient_ids_ev
                ],
            }
        )
DF_presentation = DF_presentation.dropna()
f, ax = plt.subplots(figsize=(3, 3))
sns.boxplot(
        x=feature_name,
        y="Proportion",
        data=DF_presentation,
        showfliers=False,
        order=feature_list,
        color = 'grey',
    )
annot = Annotator(
        ax,
        compare_list,
        data=DF_presentation,
        x=feature_name,
        y="Proportion",
        order=feature_list,
    )
annot.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        verbose=2,
    )
annot.apply_test()
ax, test_results = annot.annotate()
ax.set_ylabel("Proportion in each patient", fontsize=12)
ax.set_xlabel("Grade", fontsize=12)
f.savefig(
    "Results/fig16_d.jpg",
    dpi=300,
    bbox_inches="tight",
)
f.savefig(
    "Results/fig16_d.svg",
    dpi=300,
    bbox_inches="tight",
)

