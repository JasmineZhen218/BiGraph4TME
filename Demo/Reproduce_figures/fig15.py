import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from utils import preprocess_Danenberg, preprocess_Jackson
from definitions import color_palette_Bigraph, get_paired_markers
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
proportions_iv = histograms_iv / np.sum(histograms_iv, axis=1, keepdims=True)
lengths_iv = [
    survival_iv.loc[survival_iv["patientID"] == i, "time"].values[0]
    for i in patient_ids_iv
]
statuses_iv = [
    survival_iv.loc[survival_iv["patientID"] == i, "status"].values[0]
    for i in patient_ids_iv
]

threshold = 0.01
subgroup_id = 'S1'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
proportion = proportions_iv[:, np.array(characteristic_patterns)].sum(axis=1)
kmf = KaplanMeierFitter()
kmf.fit(
        np.array(lengths_iv)[proportion < threshold],
        np.array(statuses_iv)[proportion < threshold],
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
        np.array(lengths_iv)[(proportion >= threshold)],
        np.array(statuses_iv)[(proportion >= threshold)],
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
        np.array(lengths_iv)[proportion <= threshold],
        np.array(lengths_iv)[(proportion > threshold)],
        np.array(statuses_iv)[proportion <= threshold],
        np.array(statuses_iv)[(proportion > threshold)],
    )
ax.text(
        0.1,
        0.28,
        "p = {:.5f}".format(test.p_value),
        fontsize=12,
        transform=ax.transAxes,
    )
ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
ax.set_xlabel("Time (Month)", fontsize=12)
ax.set_ylabel("Survival", fontsize=12)
ax.legend(fontsize=11)
plt.show()
f.savefig("Results/fig15_a.png", dpi=300)
f.savefig("Results/fig15_a.svg", dpi=300)

subgroup_id = 'S7'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
proportion = proportions_iv[:, np.array(characteristic_patterns)].sum(axis=1)
kmf = KaplanMeierFitter()
kmf.fit(
        np.array(lengths_iv)[proportion < threshold],
        np.array(statuses_iv)[proportion < threshold],
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
        np.array(lengths_iv)[(proportion >= threshold)],
        np.array(statuses_iv)[(proportion >= threshold)],
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
        np.array(lengths_iv)[proportion <= threshold],
        np.array(lengths_iv)[(proportion > threshold)],
        np.array(statuses_iv)[proportion <= threshold],
        np.array(statuses_iv)[(proportion > threshold)],
    )
ax.text(
        0.1,
        0.28,
        "p = {:.5f}".format(test.p_value),
        fontsize=12,
        transform=ax.transAxes,
    )
ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
ax.set_xlabel("Time (Month)", fontsize=12)
ax.set_ylabel("Survival", fontsize=12)
ax.legend(fontsize=11)
plt.show()
f.savefig("Results/fig15_b.png", dpi=300)
f.savefig("Results/fig15_b.svg", dpi=300)




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
threshold = 0.01
clinical_subtype ="TNBC"
subgroup_id = 'S4'
f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
 
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1][
            "characteristic_patterns"
        ]
proportion = proportions_iv[:, np.array(characteristic_patterns)].sum(
                axis=1
            )
test = logrank_test(
                np.array(lengths_iv)[
                    (proportion < threshold)
                    & (clinical_subtypes_iv == clinical_subtype)
                ],
                np.array(lengths_iv)[
                    (proportion >= threshold)
                    & (clinical_subtypes_iv == clinical_subtype)
                ],
                np.array(statuses_iv)[
                    (proportion < threshold)
                    & (clinical_subtypes_iv == clinical_subtype)
                ],
                np.array(statuses_iv)[
                    (proportion >= threshold)
                    & (clinical_subtypes_iv == clinical_subtype)
                ],
            )
            # if test.p_value > 0.5:
            #     continue

kmf = KaplanMeierFitter()
kmf.fit(
                np.array(lengths_iv)[
                    (proportion < threshold)
                    & (clinical_subtypes_iv == clinical_subtype)
                ],
                np.array(statuses_iv)[
                    (proportion < threshold)
                    & (clinical_subtypes_iv == clinical_subtype)
                ],
                label="Negative (N = {})".format(
                    np.sum(
                        (proportion < threshold)
                        & (clinical_subtypes_iv == clinical_subtype)
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
                np.array(lengths_iv)[
                    (proportion >= threshold)
                    & (clinical_subtypes_iv == clinical_subtype)
                ],
                np.array(statuses_iv)[
                    (proportion >= threshold)
                    & (clinical_subtypes_iv == clinical_subtype)
                ],
                label="Positive (N = {})".format(
                    np.sum(
                        (proportion >= threshold)
                        & (clinical_subtypes_iv == clinical_subtype)
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
                0.28,
                "p = {:.5f}".format(test.p_value),
                fontsize=12,
                transform=ax.transAxes,
            )
ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
ax.set_xlabel("Time (Month)", fontsize=12)
ax.set_ylabel("Survival", fontsize=12)
ax.legend(fontsize=11)
f.savefig("Results/fig15_c.png", dpi=300)
f.savefig("Results/fig15_c.svg", dpi=300)


patient_ids_ev = list(SC_ev["patientID"].unique())
proportions_ev = histograms_ev / np.sum(histograms_ev, axis=1, keepdims=True)
lengths_ev = [
    survival_ev.loc[survival_ev["patientID"] == i, "time"].values[0]
    for i in patient_ids_ev
]
statuses_ev = [
    survival_ev.loc[survival_ev["patientID"] == i, "status"].values[0]
    for i in patient_ids_ev
]
threshold = 0.01
subgroup_id = 'S1'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
proportion = proportions_ev[:, np.array(characteristic_patterns)].sum(axis=1)
kmf = KaplanMeierFitter()
kmf.fit(
        np.array(lengths_ev)[proportion < threshold],
        np.array(statuses_ev)[proportion < threshold],
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
        np.array(lengths_ev)[(proportion >= threshold)],
        np.array(statuses_ev)[(proportion >= threshold)],
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
        np.array(lengths_ev)[proportion <= threshold],
        np.array(lengths_ev)[(proportion > threshold)],
        np.array(statuses_ev)[proportion <= threshold],
        np.array(statuses_ev)[(proportion > threshold)],
    )
ax.text(
        0.1,
        0.28,
        "p = {:.5f}".format(test.p_value),
        fontsize=12,
        transform=ax.transAxes,
    )
ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
ax.set_xlabel("Time (Month)", fontsize=12)
ax.set_ylabel("Survival", fontsize=12)
ax.legend(fontsize=11)
plt.show()
f.savefig("Results/fig15_d.png", dpi=300)
f.savefig("Results/fig15_d.svg", dpi=300)

subgroup_id = 'S7'
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1]["characteristic_patterns"]
f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
proportion = proportions_ev[:, np.array(characteristic_patterns)].sum(axis=1)
kmf = KaplanMeierFitter()
kmf.fit(
        np.array(lengths_ev)[proportion < threshold],
        np.array(statuses_ev)[proportion < threshold],
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
        np.array(lengths_ev)[(proportion >= threshold)],
        np.array(statuses_ev)[(proportion >= threshold)],
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
        np.array(lengths_ev)[proportion <= threshold],
        np.array(lengths_ev)[(proportion > threshold)],
        np.array(statuses_ev)[proportion <= threshold],
        np.array(statuses_ev)[(proportion > threshold)],
    )
ax.text(
        0.1,
        0.28,
        "p = {:.5f}".format(test.p_value),
        fontsize=12,
        transform=ax.transAxes,
    )
ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
ax.set_xlabel("Time (Month)", fontsize=12)
ax.set_ylabel("Survival", fontsize=12)
ax.legend(fontsize=11)
plt.show()
f.savefig("Results/fig15_e.png", dpi=300)
f.savefig("Results/fig15_e.svg", dpi=300)


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
threshold = 0.01
clinical_subtype ="TNBC"
subgroup_id = 'S4'
f, ax = plt.subplots(figsize=(3, 3.5), tight_layout=True)
 
characteristic_patterns = patient_subgroups_discovery[int(subgroup_id.split('S')[1])-1][
            "characteristic_patterns"
        ]
proportion = proportions_ev[:, np.array(characteristic_patterns)].sum(
                axis=1
            )
test = logrank_test(
                np.array(lengths_ev)[
                    (proportion < threshold)
                    & (clinical_subtypes_ev == clinical_subtype)
                ],
                np.array(lengths_ev)[
                    (proportion >= threshold)
                    & (clinical_subtypes_ev == clinical_subtype)
                ],
                np.array(statuses_ev)[
                    (proportion < threshold)
                    & (clinical_subtypes_ev == clinical_subtype)
                ],
                np.array(statuses_ev)[
                    (proportion >= threshold)
                    & (clinical_subtypes_ev == clinical_subtype)
                ],
            )
            # if test.p_value > 0.5:
            #     continue

kmf = KaplanMeierFitter()
kmf.fit(
                np.array(lengths_ev)[
                    (proportion < threshold)
                    & (clinical_subtypes_ev == clinical_subtype)
                ],
                np.array(statuses_ev)[
                    (proportion < threshold)
                    & (clinical_subtypes_ev == clinical_subtype)
                ],
                label="Negative (N = {})".format(
                    np.sum(
                        (proportion < threshold)
                        & (clinical_subtypes_ev == clinical_subtype)
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
                np.array(lengths_ev)[
                    (proportion >= threshold)
                    & (clinical_subtypes_ev == clinical_subtype)
                ],
                np.array(statuses_ev)[
                    (proportion >= threshold)
                    & (clinical_subtypes_ev == clinical_subtype)
                ],
                label="Positive (N = {})".format(
                    np.sum(
                        (proportion >= threshold)
                        & (clinical_subtypes_ev == clinical_subtype)
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
                0.28,
                "p = {:.5f}".format(test.p_value),
                fontsize=12,
                transform=ax.transAxes,
            )
ax.set(ylim=[0, 1.05], xlabel="Time (Month)", ylabel="Survival")
ax.set_xlabel("Time (Month)", fontsize=12)
ax.set_ylabel("Survival", fontsize=12)
ax.legend(fontsize=11)
f.savefig("Results/fig15_f.png", dpi=300)
f.savefig("Results/fig15_f.svg", dpi=300)