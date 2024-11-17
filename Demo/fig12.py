import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from utils import  preprocess_Danenberg
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

Histograms = bigraph_.fitted_soft_wl_subtree.Histograms
Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True)
patient_ids_discovery = np.array(list(SC_d["patientID"].unique()))


feature_name = "Grade"
feature_list = [1, 2, 3]
compare_list = [(1, 2), (2, 3), (1, 3)]

subgroup_id = 'S1'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: [
                    survival_d.loc[
                        survival_d["patientID"] == patient_id, feature_name
                    ].values[0]
                    for patient_id in patient_ids_discovery
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_a.jpg",
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
                    survival_d.loc[
                        survival_d["patientID"] == patient_id, feature_name
                    ].values[0]
                    for patient_id in patient_ids_discovery
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_e.jpg",
    dpi=300,
    bbox_inches="tight",
)

subgroup_id = 'S4'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
        {
            "Proportion": proportion[clinical_subtypes_discovery == "TNBC"],
            feature_name: [
                survival_d.loc[
                    survival_d["patientID"] == patient_id, feature_name
                ].values[0]
                for patient_id in np.array(patient_ids_discovery)[clinical_subtypes_discovery == "TNBC"]
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_i.jpg",
    dpi=300,
    bbox_inches="tight",
)



feature_name = "Tumor Stage"
feature_list = [1, 2, 3, 4]
compare_list = [
    (1, 2),
    (1,3),
    (1,4),
    (3, 4),
    (2, 3),
    (2,4)
]
subgroup_id = 'S1'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: [
                    survival_d.loc[
                        survival_d["patientID"] == patient_id, feature_name
                    ].values[0]
                    for patient_id in patient_ids_discovery
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_b.jpg",
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
                    survival_d.loc[
                        survival_d["patientID"] == patient_id, feature_name
                    ].values[0]
                    for patient_id in patient_ids_discovery
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_f.jpg",
    dpi=300,
    bbox_inches="tight",
)

subgroup_id = 'S4'
feature_list = [1, 2, 3]
compare_list = [
    (1, 2),
    (1,3),
    (2, 3),
]
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
        {
            "Proportion": proportion[clinical_subtypes_discovery == "TNBC"],
            feature_name: [
                survival_d.loc[
                    survival_d["patientID"] == patient_id, feature_name
                ].values[0]
                for patient_id in np.array(patient_ids_discovery)[clinical_subtypes_discovery == "TNBC"]
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_j.jpg",
    dpi=300,
    bbox_inches="tight",
)


feature_name = "Lymph nodes examined positive"
feature_list = ["Yes", "No"]
compare_list = [("Yes", "No")]
subgroup_id = 'S1'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: [
                    (
                        "Yes"
                        if survival_d.loc[
                            survival_d["patientID"] == patient_id, feature_name
                        ].values[0]
                        > 0
                        else "No"
                    )
                    for patient_id in patient_ids_discovery
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_c.jpg",
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
                    (
                        "Yes"
                        if survival_d.loc[
                            survival_d["patientID"] == patient_id, feature_name
                        ].values[0]
                        > 0
                        else "No"
                    )
                    for patient_id in patient_ids_discovery
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_g.jpg",
    dpi=300,
    bbox_inches="tight",
)

subgroup_id = 'S4'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion[clinical_subtypes_discovery == "TNBC"],
                feature_name: [
                    (
                        "Yes"
                        if survival_d.loc[
                            survival_d["patientID"] == patient_id, feature_name
                        ].values[0]
                        > 0
                        else "No"
                    )
                    for patient_id in np.array(patient_ids_discovery)[
                        clinical_subtypes_discovery == "TNBC"
                    ]
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_k.jpg",
    dpi=300,
    bbox_inches="tight",
)

feature_name = "Clinical Subtype"
feature_list = [
    "HER2+",
    "HR+/HER2-",
    "TNBC",
   
]
compare_list = [
    ("HER2+", "HR+/HER2-"),
    ("HER2+", "TNBC"),
    ("HR+/HER2-", "TNBC"),

]
subgroup_id = 'S1'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: clinical_subtypes_discovery

                
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_d.jpg",
    dpi=300,
    bbox_inches="tight",
)

subgroup_id = 'S7'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
proportion = np.sum(Proportions[:, np.array(characteristic_patterns)], axis=1)
DF_presentation = pd.DataFrame(
            {
                "Proportion": proportion,
                feature_name: clinical_subtypes_discovery        
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
ax.set(ylabel="Proportion in each patient")
f.savefig(
    "Results/fig12_h.jpg",
    dpi=300,
    bbox_inches="tight",
)

