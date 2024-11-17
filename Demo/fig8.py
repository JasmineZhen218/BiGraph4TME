import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import pairwise_logrank_test
from utils import preprocess_Danenberg
from definitions import (
    color_palette_Bigraph,
    color_palette_clinical,
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

patient_ids_discovery = list(SC_d["patientID"].unique())
subgroup_ids_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
subgroup_ids_discovery[:] = "Unclassified"
for i in range(len(patient_subgroups_discovery)):
    subgroup = patient_subgroups_discovery[i]
    subgroup_id = subgroup["subgroup_id"]
    patient_ids = subgroup["patient_ids"]
    subgroup_ids_discovery[np.isin(patient_ids_discovery, patient_ids)] = subgroup_id
lengths_discovery = [
    survival_d.loc[survival_d["patientID"] == i, "time"].values[0]
    for i in patient_ids_discovery
]
statuses_discovery = [
    (survival_d.loc[survival_d["patientID"] == i, "status"].values[0])
    for i in patient_ids_discovery
]
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
print(
    "{} patients in total, {} Unkonw, {} HER2+, {} HR+/HER2-, {} TNBC".format(
        len(clinical_subtypes_discovery),
        np.sum(clinical_subtypes_discovery == "Unknown"),
        np.sum(clinical_subtypes_discovery == "HER2+"),
        np.sum(clinical_subtypes_discovery == "HR+/HER2-"),
        np.sum(clinical_subtypes_discovery == "TNBC"),
    )
)

clinical_subtype = "HER2+"
subgroup_id = "S1"
lengths_discovery = np.array(lengths_discovery)
statuses_discovery = np.array(statuses_discovery)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_discovery == subgroup_id) & (
    clinical_subtypes_discovery == clinical_subtype
)
length_A, event_observed_A = (
    lengths_discovery[Indices_A],
    statuses_discovery[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_discovery != subgroup_id) & (
    clinical_subtypes_discovery == clinical_subtype
)
length_C, event_observed_C = (
    lengths_discovery[Indices_C],
    statuses_discovery[Indices_C],
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

Indices_B = clinical_subtypes_discovery != clinical_subtype
length_B, event_observed_B = (
    lengths_discovery[Indices_B],
    statuses_discovery[Indices_B],
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


groups = np.zeros_like(clinical_subtypes_discovery, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_discovery == clinical_subtype))
    & (subgroup_ids_discovery != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_discovery == clinical_subtype))
    & (subgroup_ids_discovery == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_discovery,
    groups,
    statuses_discovery,
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
f.savefig("Results/fig8_a.png", dpi=300, bbox_inches="tight")

clinical_subtype = "TNBC"
subgroup_id = "S2"
lengths_discovery = np.array(lengths_discovery)
statuses_discovery = np.array(statuses_discovery)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_discovery == subgroup_id) & (
    clinical_subtypes_discovery == clinical_subtype
)
length_A, event_observed_A = (
    lengths_discovery[Indices_A],
    statuses_discovery[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_discovery != subgroup_id) & (
    clinical_subtypes_discovery == clinical_subtype
)
length_C, event_observed_C = (
    lengths_discovery[Indices_C],
    statuses_discovery[Indices_C],
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

Indices_B = clinical_subtypes_discovery != clinical_subtype
length_B, event_observed_B = (
    lengths_discovery[Indices_B],
    statuses_discovery[Indices_B],
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


groups = np.zeros_like(clinical_subtypes_discovery, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_discovery == clinical_subtype))
    & (subgroup_ids_discovery != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_discovery == clinical_subtype))
    & (subgroup_ids_discovery == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_discovery,
    groups,
    statuses_discovery,
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
f.savefig("Results/fig8_b.png", dpi=300, bbox_inches="tight")

clinical_subtype = "HR+/HER2-"
subgroup_id = "S7"
lengths_discovery = np.array(lengths_discovery)
statuses_discovery = np.array(statuses_discovery)
f, ax = plt.subplots(figsize=(4, 4))
kmf = KaplanMeierFitter()
Indices_A = (subgroup_ids_discovery == subgroup_id) & (
    clinical_subtypes_discovery == clinical_subtype
)
length_A, event_observed_A = (
    lengths_discovery[Indices_A],
    statuses_discovery[Indices_A],
)
label = r"{} $\cap$ {} (N = {}) ".format(clinical_subtype, subgroup_id, len(length_A))
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette_Bigraph[subgroup_id],
    show_censors=True,
    censor_styles={"ms": 7, "marker": "|"},
    markerfacecolor="grey",
    linewidth=3,
    label=label,
)

Indices_C = (subgroup_ids_discovery != subgroup_id) & (
    clinical_subtypes_discovery == clinical_subtype
)
length_C, event_observed_C = (
    lengths_discovery[Indices_C],
    statuses_discovery[Indices_C],
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

Indices_B = clinical_subtypes_discovery != clinical_subtype
length_B, event_observed_B = (
    lengths_discovery[Indices_B],
    statuses_discovery[Indices_B],
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


groups = np.zeros_like(clinical_subtypes_discovery, dtype=int)
groups[:] = 1
groups[
    ((clinical_subtypes_discovery == clinical_subtype))
    & (subgroup_ids_discovery != subgroup_id)
] = 2
groups[
    ((clinical_subtypes_discovery == clinical_subtype))
    & (subgroup_ids_discovery == subgroup_id)
] = 3


log_rank_test = pairwise_logrank_test(
    lengths_discovery,
    groups,
    statuses_discovery,
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
f.savefig("Results/fig8_c.png", dpi=300, bbox_inches="tight")