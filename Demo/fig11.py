import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import  KaplanMeierFitter
from lifelines.statistics import logrank_test
from utils import preprocess_Danenberg
from definitions import color_palette_Bigraph
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

Histograms = bigraph_.fitted_soft_wl_subtree.Histograms
Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True)
lengths_discovery = [
    survival_d.loc[survival_d["patientID"] == i, "time"].values[0]
    for i in patient_ids_discovery
]
statuses_discovery = [
    (survival_d.loc[survival_d["patientID"] == i, "status"].values[0])
    for i in patient_ids_discovery
]

threshold = 0.01
numbers = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
for i in range(len(patient_subgroups_discovery)):
    subgroup_id = patient_subgroups_discovery[i]["subgroup_id"]
    characteristic_patterns = patient_subgroups_discovery[i]["characteristic_patterns"]
    f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
    proportion = Proportions[:, np.array(characteristic_patterns)].sum(axis=1)
    kmf = KaplanMeierFitter()
    kmf.fit(
        np.array(lengths_discovery)[proportion < threshold],
        np.array(statuses_discovery)[proportion < threshold],
        label="Negative (N = {})".format(np.sum(proportion < threshold)),
    )
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        show_censors=True,
        censor_styles={"ms": 6, "marker": "|"},
        color="grey",
        linewidth=3,
    )
    kmf.fit(
        np.array(lengths_discovery)[(proportion >= threshold)],
        np.array(statuses_discovery)[(proportion >= threshold)],
        label="Positive (N = {})".format(np.sum((proportion >= threshold))),
    )
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        show_censors=True,
        censor_styles={"ms": 6, "marker": "|"},
        color=color_palette_Bigraph[subgroup_id],
        linewidth=3,
    )
    test = logrank_test(
        np.array(lengths_discovery)[(proportion < threshold)],
        np.array(lengths_discovery)[(proportion >= threshold)],
        np.array(statuses_discovery)[proportion < threshold],
        np.array(statuses_discovery)[(proportion >= threshold)],
    )
    ax.text(
        0.1, 0.3, "p = {:.5f}".format(test.p_value), fontsize=12, transform=ax.transAxes
    )
    ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
    ax.set_xlabel("Time (Month)", fontsize=12)
    ax.set_ylabel("Survival", fontsize=12)
    ax.legend(fontsize=11)
    f.savefig("Results/fig11_{}.jpg".format(numbers[i]), dpi=300, bbox_inches="tight")


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
threshold = 0.01
for clinical_subtype in [ "TNBC"]:
    for i in range(len(patient_subgroups_discovery)):
        subgroup_id = patient_subgroups_discovery[i]["subgroup_id"]
        if subgroup_id != 'S4':
            continue
        characteristic_patterns = patient_subgroups_discovery[i][
            "characteristic_patterns"
        ]
        f, ax = plt.subplots(
        1,
        1,
        figsize=(3, 3.5),
        tight_layout=True,
    )
        proportion = Proportions[:, np.array(characteristic_patterns)].sum(axis=1)
        test = logrank_test(
            np.array(lengths_discovery)[
                (proportion < threshold)
                & (clinical_subtypes_discovery == clinical_subtype)
            ],
            np.array(lengths_discovery)[
                (proportion >= threshold)
                & (clinical_subtypes_discovery == clinical_subtype)
            ],
            np.array(statuses_discovery)[
                (proportion < threshold)
                & (clinical_subtypes_discovery == clinical_subtype)
            ],
            np.array(statuses_discovery)[
                (proportion >= threshold)
                & (clinical_subtypes_discovery == clinical_subtype)
            ],
        )
        # if test.p_value > 0.5:
        #     continue

        kmf = KaplanMeierFitter()

        kmf.fit(
            np.array(lengths_discovery)[
                (proportion < threshold)
                & (clinical_subtypes_discovery == clinical_subtype)
            ],
            np.array(statuses_discovery)[
                (proportion < threshold)
                & (clinical_subtypes_discovery == clinical_subtype)
            ],
            label="Negative (N = {})".format(
                np.sum(
                    (proportion < threshold)
                    & (clinical_subtypes_discovery == clinical_subtype)
                )
            ),
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={"ms": 6, "marker": "|"},
            color="grey",
            linewidth=3,
        )
        kmf.fit(
            np.array(lengths_discovery)[
                (proportion >= threshold)
                & (clinical_subtypes_discovery == clinical_subtype)
            ],
            np.array(statuses_discovery)[
                (proportion >= threshold)
                & (clinical_subtypes_discovery == clinical_subtype)
            ],
            label="Positive (N = {})".format(
                np.sum(
                    (proportion >= threshold)
                    & (clinical_subtypes_discovery == clinical_subtype)
                )
            ),
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={"ms": 6, "marker": "|"},
            color=color_palette_Bigraph[subgroup_id],
            linewidth=3,
        )

        ax.text(
            0.1,
            0.3,
            "p = {:.5f}".format(test.p_value),
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
        ax.set_xlabel("Time (Month)", fontsize=12)
        ax.set_ylabel("Survival", fontsize=12)
        # ax.set_title(
        #     "Clinical Subtype: {}, Subgroup: {}".format(clinical_subtype, subgroup_id),
        #     fontsize=12,
        # )
        ax.legend(fontsize=11)
    plt.show()
    f.savefig("Results/fig11_h.jpg", dpi=300, bbox_inches="tight")
