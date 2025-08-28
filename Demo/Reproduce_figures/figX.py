import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import seaborn as sns
sys.path.append("./..")
sys.path.append("./../..")
from utils import preprocess_Danenberg
from bi_graph import BiGraph
from definitions import Cell_types_displayed_Danenberg
    
SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)

patient_ids_discovery = list(SC_d["patientID"].unique())
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

from definitions import color_palette_clinical
kmf = KaplanMeierFitter()
f, ax = plt.subplots(figsize=(5, 5))
for clinical_subgroup in ['HR+/HER2-', 'HER2+', 'TNBC', 'Unknown']:
    length_A, event_observed_A = (
        np.array(lengths_discovery)[clinical_subtypes_discovery == clinical_subgroup],
        np.array(statuses_discovery)[clinical_subtypes_discovery == clinical_subgroup],
    )
    label = "{} (N={})".format(
        clinical_subgroup, np.sum(clinical_subtypes_discovery == clinical_subgroup)
    )
    kmf.fit(length_A, event_observed_A, label=label)
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        color=color_palette_clinical[clinical_subgroup],
        show_censors=True,
        linewidth=2,
        censor_styles={"ms": 5, "marker": "|"},
    )
clinical_subtypes_discovery_ids = np.zeros(len(patient_ids_discovery), dtype=object)
for i in range(len(clinical_subtypes_discovery)):
    if clinical_subtypes_discovery[i] == "HR+/HER2-":
        clinical_subtypes_discovery_ids[i] = 0
    elif clinical_subtypes_discovery[i] == "HER2+":
        clinical_subtypes_discovery_ids[i] = 1
    elif clinical_subtypes_discovery[i] == "TNBC":
        clinical_subtypes_discovery_ids[i] = 2
    elif clinical_subtypes_discovery[i] == "Unknown":
        clinical_subtypes_discovery_ids[i] = 3
    
log_rank_test = multivariate_logrank_test(
    np.array(lengths_discovery),
    np.array(clinical_subtypes_discovery),
    np.array(statuses_discovery),
)
p_value = log_rank_test.p_value
ax.legend(ncol=2, fontsize=10)
ax.text(
    x=0.3,
    y=0.95,
    s="p-value = {:.5f}".format(p_value),
    fontsize=14,
    transform=ax.transAxes,
)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
sns.despine()
f.savefig("Results/figX_clinical_subtype.svg", dpi=600, bbox_inches="tight")

color_palette_stage = {
    "Unknown": "grey",
    "Stage I": sns.color_palette("Set2")[1],
    "Stage II": sns.color_palette("Set2")[4],
    "Stage III": sns.color_palette("Set2")[2],
    "Stage IV": sns.color_palette("Set2")[3],
}
stages_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
stages_discovery[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    stage = survival_d.loc[survival_d["patientID"] == patient_id, "Tumor Stage"].values[
        0
    ]
    if stage == 1.0:
        stages_discovery[i] = "Stage I"
    elif stage == 2.0:
        stages_discovery[i] = "Stage II"
    elif stage == 3.0:
        stages_discovery[i] = "Stage III"
    elif stage == 4.0:
        stages_discovery[i] = "Stage IV"
stages_discovery_ids = np.zeros(len(patient_ids_discovery), dtype=object)
for i in range(len(stages_discovery)):
    if stages_discovery[i] == "Stage I":
        stages_discovery_ids[i] = 0
    elif stages_discovery[i] == "Stage II":
        stages_discovery_ids[i] = 1
    elif stages_discovery[i] == "Stage III":
        stages_discovery_ids[i] = 2
    elif stages_discovery[i] == "Stage IV":
        stages_discovery_ids[i] = 3
    elif stages_discovery[i] == "Unknown":
        stages_discovery_ids[i] = 4
f, ax = plt.subplots(figsize=(5, 5))
for stage in ["Stage I", "Stage II", "Stage III", "Stage IV", "Unknown"]:
    length_A, event_observed_A = (
        np.array(lengths_discovery)[stages_discovery == stage],
        np.array(statuses_discovery)[stages_discovery == stage],
    )
    label = "{} (N={})".format(
        stage, np.sum(stages_discovery == stage)
    )
    kmf.fit(length_A, event_observed_A, label=label)
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        color=color_palette_stage[stage],
        show_censors=True,
        linewidth=2,
        censor_styles={"ms": 5, "marker": "|"},
    )
log_rank_test = multivariate_logrank_test(
    np.array(lengths_discovery),
    np.array(stages_discovery),
    np.array(statuses_discovery),
)
p_value = log_rank_test.p_value
ax.legend(ncol=2, fontsize=10)
ax.text(
    x=0.3,
    y=0.95,
    s="p-value = {:.5f}".format(p_value),
    fontsize=14,
    transform=ax.transAxes,
)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
sns.despine()
f.savefig("Results/figX_stage.svg", dpi=600, bbox_inches="tight")


color_palette_grade = {
    "Unknown": "grey",
    "Grade 1": sns.color_palette("Set2")[0],
    "Grade 2": sns.color_palette("Set2")[1],
    "Grade 3": sns.color_palette("Set2")[2],
}
grades_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
grades_discovery[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    grade = survival_d.loc[survival_d["patientID"] == patient_id, "Grade"].values[0]
    if grade == 1.0:
        grades_discovery[i] = "Grade 1"
    elif grade == 2.0:
        grades_discovery[i] = "Grade 2"
    elif grade == 3.0:
        grades_discovery[i] = "Grade 3"

grades_discovery_ids = np.zeros(len(patient_ids_discovery), dtype=object)
for i in range(len(grades_discovery)):
    if grades_discovery[i] == "Grade 1":
        grades_discovery_ids[i] = 0
    elif grades_discovery[i] == "Grade 2":
        grades_discovery_ids[i] = 1
    elif grades_discovery[i] == "Grade 3":
        grades_discovery_ids[i] = 2
    elif grades_discovery[i] == "Unknown":
        grades_discovery_ids[i] = 3
f, ax = plt.subplots(figsize=(5, 5))
for grade in ["Grade 1", "Grade 2", "Grade 3", "Unknown"]:
    length_A, event_observed_A = (
        np.array(lengths_discovery)[grades_discovery == grade],
        np.array(statuses_discovery)[grades_discovery == grade],
    )
    label = "{} (N={})".format(
        grade, np.sum(grades_discovery == grade)
    )
    kmf.fit(length_A, event_observed_A, label=label)
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        color=color_palette_grade[grade],
        show_censors=True,
        linewidth=2,
        censor_styles={"ms": 5, "marker": "|"},
    )
log_rank_test = multivariate_logrank_test(
    np.array(lengths_discovery),
    np.array(grades_discovery),
    np.array(statuses_discovery),
)
p_value = log_rank_test.p_value
ax.legend(ncol=2, fontsize=10)
ax.text(
    x=0.3,
    y=0.95,
    s="p-value = {:.5f}".format(p_value),
    fontsize=14,
    transform=ax.transAxes,
)
ax.set_xlabel("Time (Month)", fontsize=14)
ax.set_ylabel("Cumulative Survival", fontsize=14)
ax.set(
    ylim=(-0.05, 1.05),
)
sns.despine()
f.savefig("Results/figX_grade.svg", dpi=600, bbox_inches="tight")