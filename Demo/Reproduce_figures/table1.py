import sys
import numpy as np
import pandas as pd
from utils import preprocess_Danenberg
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
print(
    "{} patients in total, {} Unkonw, {} Stage I, {} Stage II, {} Stage III, {} Stage IV".format(
        len(stages_discovery),
        np.sum(stages_discovery == "Unknown"),
        np.sum(stages_discovery == "Stage I"),
        np.sum(stages_discovery == "Stage II"),
        np.sum(stages_discovery == "Stage III"),
        np.sum(stages_discovery == "Stage IV"),
    )
)

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

print(
    "{} patients in total, {} Unkonw, {} Grade 1, {} Grade 2, {} Grade 3".format(
        len(grades_discovery),
        np.sum(grades_discovery == "Unknown"),
        np.sum(grades_discovery == "Grade 1"),
        np.sum(grades_discovery == "Grade 2"),
        np.sum(grades_discovery == "Grade 3"),
    )
)

positive_lymph = np.zeros(len(patient_ids_discovery), dtype=object)
positive_lymph[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    lymph = survival_d.loc[
        survival_d["patientID"] == patient_id, "LymphNodesOrdinal"
    ].values[0]
    positive_lymph[i] = lymph

print(
    "{} patients in total, {} Unkonw, {} 0, {} 1, {} 2-3, {} 4-7, {} 7+".format(
        len(positive_lymph),
        np.sum(positive_lymph == "Unknown"),
        np.sum(positive_lymph == "0"),
        np.sum(positive_lymph == "1"),
        np.sum(positive_lymph == "2-3"),
        np.sum(positive_lymph == "4-7"),
        np.sum(positive_lymph == "7+"),
    )
)

ages_discovery = np.zeros(len(patient_ids_discovery), dtype=object)
ages_discovery[:] = "Unknown"
for i in range(len(patient_ids_discovery)):
    patient_id = patient_ids_discovery[i]
    age = survival_d.loc[
        survival_d["patientID"] == patient_id, "Age at Diagnosis"
    ].values[0]
    if age < 50:
        ages_discovery[i] = "<50"
    elif age >= 50 and age <= 70:
        ages_discovery[i] = "50-70"
    elif age > 70:
        ages_discovery[i] = ">70"
print(
    "{} patients in total, {} Unkonw, {} <50, {} 50-70, {} >70".format(
        len(ages_discovery),
        np.sum(ages_discovery == "Unknown"),
        np.sum(ages_discovery == "<50"),
        np.sum(ages_discovery == "50-70"),
        np.sum(ages_discovery == ">70"),
    )
)

def caculate_mutual_similarity(gram_matrix):
    gram_matrix_ = gram_matrix.copy()
    np.fill_diagonal(gram_matrix_, -1)
    return np.mean(gram_matrix_[gram_matrix_ != -1])


Similarity_matrix = bigraph_.Similarity_matrix
print(
    "The mutual similarity of the population graph is {:.2f}".format(
        caculate_mutual_similarity(Similarity_matrix)
    )
)

Gram_matrix_ = Similarity_matrix[
    np.array(np.where(clinical_subtypes_discovery == "HER2+")[0].tolist()), :
][:, np.array(np.where(clinical_subtypes_discovery == "HER2+")[0].tolist())]
print(
    "{} HER2+ patients, averaged Intra-group similarity is {:.2f} ".format(
        Gram_matrix_.shape[0], caculate_mutual_similarity(Gram_matrix_)
    )
)

Gram_matrix_ = Similarity_matrix[
    np.array(np.where(clinical_subtypes_discovery == "HR+/HER2-")[0].tolist()), :
][:, np.array(np.where(clinical_subtypes_discovery == "HR+/HER2-")[0].tolist())]
print(
    "{} HR+/HER2- patients, averaged Intra-group similarity is {:.2f} ".format(
        Gram_matrix_.shape[0], caculate_mutual_similarity(Gram_matrix_)
    )
)

Gram_matrix_ = Similarity_matrix[
    np.array(np.where(clinical_subtypes_discovery == "TNBC")[0].tolist()), :
][:, np.array(np.where(clinical_subtypes_discovery == "TNBC")[0].tolist())]
print(
    "{} TNBC patients, averaged Intra-group similarity is {:.2f} ".format(
        Gram_matrix_.shape[0], caculate_mutual_similarity(Gram_matrix_)
    )
)

print("\n")

for stage in ["Stage I", "Stage II", "Stage III", "Stage IV"]:
    Gram_matrix_ = Similarity_matrix[
        np.array(np.where(stages_discovery == stage)[0].tolist()), :
    ][:, np.array(np.where(stages_discovery == stage)[0].tolist())]
    print(
        "{} Stage {} patients, averaged Intra-group similarity is {:.2f} ".format(
            Gram_matrix_.shape[0], stage, caculate_mutual_similarity(Gram_matrix_)
        )
    )

print("\n")
for grade in ["Grade 1", "Grade 2", "Grade 3"]:
    Gram_matrix_ = Similarity_matrix[
        np.array(np.where(grades_discovery == grade)[0].tolist()), :
    ][:, np.array(np.where(grades_discovery == grade)[0].tolist())]
    print(
        "{} Grade {} patients, averaged Intra-group similarity is {:.2f} ".format(
            Gram_matrix_.shape[0], grade, caculate_mutual_similarity(Gram_matrix_)
        )
    )

print("\n")
for lymph in ["0", "1", "2-3", "4-7", "7+"]:
    Gram_matrix_ = Similarity_matrix[
        np.array(np.where(positive_lymph == lymph)[0].tolist()), :
    ][:, np.array(np.where(positive_lymph == lymph)[0].tolist())]
    print(
        "{} patients with {} positive lymph nodes, averaged Intra-group similarity is {:.2f} ".format(
            Gram_matrix_.shape[0], lymph, caculate_mutual_similarity(Gram_matrix_)
        )
    )

print("\n")
for age in ["<50", "50-70", ">70"]:
    Gram_matrix_ = Similarity_matrix[
        np.array(np.where(ages_discovery == age)[0].tolist()), :
    ][:, np.array(np.where(ages_discovery == age)[0].tolist())]
    print(
        "{} patients with age {}, averaged Intra-group similarity is {:.2f} ".format(
            Gram_matrix_.shape[0], age, caculate_mutual_similarity(Gram_matrix_)
        )
    )

intra_cluster_similarity = 0
for subgroup_id in [
    "S" + str(i) for i in range(1, len(patient_subgroups_discovery) + 1)
]:
    Gram_matrix_ = Similarity_matrix[
        np.array(np.where(subgroup_ids_discovery == subgroup_id)[0].tolist()), :
    ][:, np.array(np.where(subgroup_ids_discovery == subgroup_id)[0].tolist())]
    intra_cluster_similarity += np.sum(Gram_matrix_)
    print(
        "{} patients in {}, averaged Intra-group similarity is {:.2f} ".format(
            subgroup_id, Gram_matrix_.shape[0], caculate_mutual_similarity(Gram_matrix_)
        )
    )
inter_cluster_similarity = np.sum(Similarity_matrix) - intra_cluster_similarity
print(
    "Intra-cluster similarity is {:.2f}, inter-cluster similarity = {:.2f}, ratio = {:.2f}".format(
        intra_cluster_similarity,
        inter_cluster_similarity,
        intra_cluster_similarity / inter_cluster_similarity,
    )
)
