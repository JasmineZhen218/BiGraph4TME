import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import (
    pairwise_logrank_test,
)
from utils import preprocess_Danenberg, preprocess_Jackson
from definitions import (
    color_palette_Bigraph,
    color_palette_clinical,
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

patient_ids_iv = list(SC_iv["patientID"].unique())
subgroup_ids_iv = np.zeros(len(patient_ids_iv), dtype=object)
subgroup_ids_iv[:] = "Unclassified"
for i in range(len(patient_subgroups_iv)):
    subgroup = patient_subgroups_iv[i]
    subgroup_id = subgroup["subgroup_id"]
    patient_ids = subgroup["patient_ids"]
    subgroup_ids_iv[np.isin(patient_ids_iv, patient_ids)] = subgroup_id

lengths_iv = [
    survival_iv.loc[survival_iv["patientID"] == i, "time"].values[0]
    for i in patient_ids_iv
]
statuses_iv = [
    survival_iv.loc[survival_iv["patientID"] == i, "status"].values[0]
    for i in patient_ids_iv
]

clinical_subtypes_iv = np.zeros(len(patient_ids_iv), dtype=object)
clinical_subtypes_iv[:] = "Unknown"
for i in range(len(patient_ids_iv)):
    patient_id = patient_ids_iv[i]
    er = survival_iv.loc[survival_iv["patientID"] == patient_id, "ER Status"].values[0]
    pr = survival_iv.loc[survival_iv["patientID"] == patient_id, "PR Status"].values[0]
    her2 = survival_iv.loc[
        survival_iv["patientID"] == patient_id, "HER2 Status"
    ].values[0]
    if her2 == "Positive":
        clinical_subtypes_iv[i] = "HER2+"  # Her2+
    if (er == "Positive" or pr == "Positive") and her2 == "Negative":
        clinical_subtypes_iv[i] = "HR+/HER2-"  # HR+/HER2-
    elif (er == "Negative" and pr == "Negative") and her2 == "Negative":
        clinical_subtypes_iv[i] = "TNBC"  # TNBC
print(
    "{} patients in total, {} Unkonw, {} HER2+, {} HR+/HER2-, {} TNBC".format(
        len(clinical_subtypes_iv),
        np.sum(clinical_subtypes_iv == "Unknown"),
        np.sum(clinical_subtypes_iv == "HER2+"),
        np.sum(clinical_subtypes_iv == "HR+/HER2-"),
        np.sum(clinical_subtypes_iv == "TNBC"),
    )
)
clinical_subtype = "HER2+"
subgroup_id = "S1'"
lengths_iv = np.array(lengths_iv)
statuses_iv = np.array(statuses_iv)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_iv == subgroup_id) & (
    clinical_subtypes_iv == clinical_subtype
)
length_A, event_observed_A = (
    lengths_iv[Indices_A],
    statuses_iv[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id[:-1]],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_iv != subgroup_id) & (
    clinical_subtypes_iv == clinical_subtype
)
length_C, event_observed_C = (
    lengths_iv[Indices_C],
    statuses_iv[Indices_C],
)
label = r"{} \  {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_C))
kmf.fit(length_C, event_observed_C, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_clinical[clinical_subtype],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
)

Indices_B = clinical_subtypes_iv != clinical_subtype
length_B, event_observed_B = (
    lengths_iv[Indices_B],
    statuses_iv[Indices_B],
)
label = r"Others (N = {})".format(len(length_B))
kmf.fit(length_B, event_observed_B, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color="grey",
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    linewidth=3,
    label=label,
)


groups = np.zeros_like(clinical_subtypes_iv, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_iv == clinical_subtype)) & (subgroup_ids_iv != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_iv == clinical_subtype)) & (subgroup_ids_iv == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_iv,
    groups,
    statuses_iv,
)
print(log_rank_test.summary)


ax.legend(fontsize=12)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
ax.set_title(
    "Intersection of {} and {}".format(clinical_subtype, subgroup_id), fontsize=14
)
f.savefig(
    "Results/fig14_a.jpg",
    dpi=300,
    bbox_inches="tight",
)


clinical_subtype = "TNBC"
subgroup_id = "S2'"
lengths_iv = np.array(lengths_iv)
statuses_iv = np.array(statuses_iv)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_iv == subgroup_id) & (
    clinical_subtypes_iv == clinical_subtype
)
length_A, event_observed_A = (
    lengths_iv[Indices_A],
    statuses_iv[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id[:-1]],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_iv != subgroup_id) & (
    clinical_subtypes_iv == clinical_subtype
)
length_C, event_observed_C = (
    lengths_iv[Indices_C],
    statuses_iv[Indices_C],
)
label = r"{} \  {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_C))
kmf.fit(length_C, event_observed_C, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_clinical[clinical_subtype],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
)

Indices_B = clinical_subtypes_iv != clinical_subtype
length_B, event_observed_B = (
    lengths_iv[Indices_B],
    statuses_iv[Indices_B],
)
label = r"Others (N = {})".format(len(length_B))
kmf.fit(length_B, event_observed_B, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color="grey",
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    linewidth=3,
    label=label,
)


groups = np.zeros_like(clinical_subtypes_iv, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_iv == clinical_subtype)) & (subgroup_ids_iv != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_iv == clinical_subtype)) & (subgroup_ids_iv == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_iv,
    groups,
    statuses_iv,
)
print(log_rank_test.summary)


ax.legend(fontsize=12)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
ax.set_title(
    "Intersection of {} and {}".format(clinical_subtype, subgroup_id), fontsize=14
)
plt.show()
f.savefig(
    "Results/fig14_b.jpg",
    dpi=300,
    bbox_inches="tight",
)


clinical_subtype = "HR+/HER2-"
subgroup_id = "S7'"
lengths_iv = np.array(lengths_iv)
statuses_iv = np.array(statuses_iv)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_iv == subgroup_id) & (
    clinical_subtypes_iv == clinical_subtype
)
length_A, event_observed_A = (
    lengths_iv[Indices_A],
    statuses_iv[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id[:-1]],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_iv != subgroup_id) & (
    clinical_subtypes_iv == clinical_subtype
)
length_C, event_observed_C = (
    lengths_iv[Indices_C],
    statuses_iv[Indices_C],
)
label = r"{} \  {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_C))
kmf.fit(length_C, event_observed_C, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_clinical[clinical_subtype],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
)

Indices_B = clinical_subtypes_iv != clinical_subtype
length_B, event_observed_B = (
    lengths_iv[Indices_B],
    statuses_iv[Indices_B],
)
label = r"Others (N = {})".format(len(length_B))
kmf.fit(length_B, event_observed_B, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color="grey",
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    linewidth=3,
    label=label,
)


groups = np.zeros_like(clinical_subtypes_iv, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_iv == clinical_subtype)) & (subgroup_ids_iv != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_iv == clinical_subtype)) & (subgroup_ids_iv == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_iv,
    groups,
    statuses_iv,
)
print(log_rank_test.summary)


ax.legend(fontsize=12)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
ax.set_title(
    "Intersection of {} and {}".format(clinical_subtype, subgroup_id), fontsize=14
)
plt.show()
f.savefig(
    "Results/fig14_c.jpg",
    dpi=300,
    bbox_inches="tight",
)

patient_ids_ev = list(SC_ev["patientID"].unique())
subgroup_ids_ev = np.zeros(len(patient_ids_ev), dtype=object)
subgroup_ids_ev[:] = "Unclassified"
for i in range(len(patient_subgroups_ev)):
    subgroup = patient_subgroups_ev[i]
    subgroup_id = subgroup["subgroup_id"]
    patient_ids = subgroup["patient_ids"]
    subgroup_ids_ev[np.isin(patient_ids_ev, patient_ids)] = subgroup_id

lengths_ev = [
    survival_ev.loc[survival_ev["patientID"] == i, "time"].values[0]
    for i in patient_ids_ev
]
statuses_ev = [
    survival_ev.loc[survival_ev["patientID"] == i, "status"].values[0]
    for i in patient_ids_ev
]

clinical_subtypes_ev = np.zeros(len(patient_ids_ev), dtype=object)
clinical_subtypes_ev[:] = "Unknown"
for i in range(len(patient_ids_ev)):
    patient_id = patient_ids_ev[i]
    er = survival_ev.loc[survival_ev["patientID"] == patient_id, "ERStatus"].values[0]
    pr = survival_ev.loc[survival_ev["patientID"] == patient_id, "PRStatus"].values[0]
    her2 = survival_ev.loc[survival_ev["patientID"] == patient_id, "HER2Status"].values[
        0
    ]
    if her2 == "positive":
        clinical_subtypes_ev[i] = "HER2+"  # Her2+
    if (er == "positive" or pr == "positive") and her2 == "negative":
        clinical_subtypes_ev[i] = "HR+/HER2-"  # HR+/HER2-
    elif (er == "negative" and pr == "negative") and her2 == "negative":
        clinical_subtypes_ev[i] = "TNBC"  # TNBC
print(
    "{} patients in total, {} Unkonw, {} HER2+, {} HR+/HER2-, {} TNBC".format(
        len(clinical_subtypes_iv),
        np.sum(clinical_subtypes_ev == "Unknown"),
        np.sum(clinical_subtypes_ev == "HER2+"),
        np.sum(clinical_subtypes_ev == "HR+/HER2-"),
        np.sum(clinical_subtypes_ev == "TNBC"),
    )
)
clinical_subtype = "HER2+"
subgroup_id = "S1'"
lengths_ev = np.array(lengths_ev)
statuses_ev = np.array(statuses_ev)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_ev == subgroup_id) & (
    clinical_subtypes_ev == clinical_subtype
)
length_A, event_observed_A = (
    lengths_ev[Indices_A],
    statuses_ev[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id[:-1]],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_ev != subgroup_id) & (
    clinical_subtypes_ev == clinical_subtype
)
length_C, event_observed_C = (
    lengths_ev[Indices_C],
    statuses_ev[Indices_C],
)
label = r"{} \  {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_C))
kmf.fit(length_C, event_observed_C, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_clinical[clinical_subtype],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
)

Indices_B = clinical_subtypes_ev != clinical_subtype
length_B, event_observed_B = (
    lengths_ev[Indices_B],
    statuses_ev[Indices_B],
)
label = r"Others (N = {})".format(len(length_B))
kmf.fit(length_B, event_observed_B, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color="grey",
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    linewidth=3,
    label=label,
)


groups = np.zeros_like(clinical_subtypes_ev, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_ev == clinical_subtype)) & (subgroup_ids_ev != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_ev == clinical_subtype)) & (subgroup_ids_ev == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_ev,
    groups,
    statuses_ev,
)
print(log_rank_test.summary)


ax.legend(fontsize=12)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
ax.set_title(
    "Intersection of {} and {}".format(clinical_subtype, subgroup_id), fontsize=14
)
plt.show()
f.savefig(
    "Results/fig14_d.jpg",
    dpi=300,
    bbox_inches="tight",
)

clinical_subtype = "TNBC"
subgroup_id = "S2'"
lengths_ev = np.array(lengths_ev)
statuses_ev = np.array(statuses_ev)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_ev == subgroup_id) & (
    clinical_subtypes_ev == clinical_subtype
)
length_A, event_observed_A = (
    lengths_ev[Indices_A],
    statuses_ev[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id[:-1]],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_ev != subgroup_id) & (
    clinical_subtypes_ev == clinical_subtype
)
length_C, event_observed_C = (
    lengths_ev[Indices_C],
    statuses_ev[Indices_C],
)
label = r"{} \  {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_C))
kmf.fit(length_C, event_observed_C, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_clinical[clinical_subtype],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
)

Indices_B = clinical_subtypes_ev != clinical_subtype
length_B, event_observed_B = (
    lengths_ev[Indices_B],
    statuses_ev[Indices_B],
)
label = r"Others (N = {})".format(len(length_B))
kmf.fit(length_B, event_observed_B, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color="grey",
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    linewidth=3,
    label=label,
)


groups = np.zeros_like(clinical_subtypes_ev, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_ev == clinical_subtype)) & (subgroup_ids_ev != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_ev == clinical_subtype)) & (subgroup_ids_ev == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_ev,
    groups,
    statuses_ev,
)
print(log_rank_test.summary)


ax.legend(fontsize=12)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
ax.set_title(
    "Intersection of {} and {}".format(clinical_subtype, subgroup_id), fontsize=14
)
plt.show()
f.savefig(
    "Results/fig14_e.jpg",
    dpi=300,
    bbox_inches="tight",
)

clinical_subtype = "HR+/HER2-"
subgroup_id = "S7'"
lengths_ev = np.array(lengths_ev)
statuses_ev = np.array(statuses_ev)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_ev == subgroup_id) & (
    clinical_subtypes_ev == clinical_subtype
)
length_A, event_observed_A = (
    lengths_ev[Indices_A],
    statuses_ev[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id[:-1]],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_ev != subgroup_id) & (
    clinical_subtypes_ev == clinical_subtype
)
length_C, event_observed_C = (
    lengths_ev[Indices_C],
    statuses_ev[Indices_C],
)
label = r"{} \  {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_C))
kmf.fit(length_C, event_observed_C, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_clinical[clinical_subtype],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
)

Indices_B = clinical_subtypes_ev != clinical_subtype
length_B, event_observed_B = (
    lengths_ev[Indices_B],
    statuses_ev[Indices_B],
)
label = r"Others (N = {})".format(len(length_B))
kmf.fit(length_B, event_observed_B, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color="grey",
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    linewidth=3,
    label=label,
)


groups = np.zeros_like(clinical_subtypes_ev, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_ev == clinical_subtype)) & (subgroup_ids_ev != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_ev == clinical_subtype)) & (subgroup_ids_ev == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_ev,
    groups,
    statuses_ev,
)
print(log_rank_test.summary)


ax.legend(fontsize=12)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
ax.set_title(
    "Intersection of {} and {}".format(clinical_subtype, subgroup_id), fontsize=14
)
plt.show()
f.savefig(
    "Results/fig14_f.jpg",
    dpi=300,
    bbox_inches="tight",
)