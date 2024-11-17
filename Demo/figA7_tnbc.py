import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from utils import preprocess_Danenberg, preprocess_Jackson, preprocess_Wang
from definitions import (
    get_paired_markers
)
sys.path.append("./..")
from bi_graph import BiGraph


SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
SC_ev_raw = pd.read_csv("Datasets/Jackson_et_al/cells.csv")
survival_ev_raw = pd.read_csv("Datasets/Jackson_et_al/clinical.csv")
SC_ev, survival_ev = preprocess_Jackson(SC_ev_raw, survival_ev_raw)
SC_ev_tnbc_raw = pd.read_csv("Datasets/Wang_TNBC/cells.csv")
survival_ev_tnbc_raw = pd.read_csv("Datasets/Wang_TNBC/clinical.csv")
SC_ev_tnbc, survival_ev_tnbc = preprocess_Wang(SC_ev_tnbc_raw, survival_ev_tnbc_raw)

bigraph_ = BiGraph(k_patient_clustering=30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, survival_data=survival_d
)
population_graph_iv, patient_subgroups_iv, histograms_iv, Signature_iv = bigraph_.transform(
    SC_iv, survival_data=survival_iv
)


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
Signatures = bigraph_.fitted_soft_wl_subtree.Signatures
Histograms = bigraph_.fitted_soft_wl_subtree.Histograms
Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True)
threshold = 0.5  # does not impact the downstream analysis, only imapct the presentation of the signature map
tumor_niche = np.where(
    (np.sum(Signatures[:, :16] > threshold, axis=1) > 0)
    & (np.sum(Signatures[:, 16:] > threshold, axis=1) == 0)
)[0]
immune_niche = np.where(
    (np.sum(Signatures[:, :16] > threshold, axis=1) == 0)
    & (np.sum(Signatures[:, 16:27] > threshold, axis=1) > 0)
    & (np.sum(Signatures[:, 27:] > threshold, axis=1) == 0)
)[0]
stromal_niche = np.where(
    (np.sum(Signatures[:, :16] > threshold, axis=1) == 0)
    & (np.sum(Signatures[:, 16:27] > threshold, axis=1) == 0)
    & (np.sum(Signatures[:, 27:] > threshold, axis=1) > 0)
)[0]
interacting_niche = [
    i
    for i in range(Signatures.shape[0])
    if i not in np.concatenate([tumor_niche, immune_niche, stromal_niche])
]
print("There are {} identified TME patterns.".format(Signatures.shape[0]))
print(
    "There are {} tumor niches, {} immune niches, {} stromal niches, and {} interacting niches.".format(
        len(tumor_niche), len(immune_niche), len(stromal_niche), len(interacting_niche)
    )
)
tme_pattern_orders = np.concatenate(
    [tumor_niche, immune_niche, stromal_niche, interacting_niche]
)

DF_proportion = pd.DataFrame(Proportions)
DF_proportion = DF_proportion.melt(var_name="pattern_id", value_name="proportion")
DF_proportion["pattern_id"] = DF_proportion["pattern_id"].astype(str)

DF_proportion["dataset"] = "Discovery set"
Proportions_iv = histograms_iv / np.sum(histograms_iv, axis=1, keepdims=True)
DF_proportion_iv = pd.DataFrame(Proportions_iv)
DF_proportion_iv = DF_proportion_iv.melt(var_name="pattern_id", value_name="proportion")
DF_proportion_iv["pattern_id"] = DF_proportion_iv["pattern_id"].astype(str)
DF_proportion_iv["dataset"] = "Inner Validation set"
Proportions_ev = histograms_ev / np.sum(histograms_ev, axis=1, keepdims=True)
DF_proportion_ev = pd.DataFrame(Proportions_ev)
DF_proportion_ev = DF_proportion_ev.melt(var_name="pattern_id", value_name="proportion")
DF_proportion_ev["pattern_id"] = DF_proportion_ev["pattern_id"].astype(str)
DF_proportion_ev["dataset"] = "External validation set-1"
Proportions_ev_tnbc_baseline = histograms_ev_tnbc_baseline / np.sum(histograms_ev_tnbc_baseline, axis=1, keepdims=True)
DF_proportion_ev_tnbc_baseline = pd.DataFrame(Proportions_ev_tnbc_baseline)
DF_proportion_ev_tnbc_baseline = DF_proportion_ev_tnbc_baseline.melt(var_name="pattern_id", value_name="proportion")
DF_proportion_ev_tnbc_baseline["pattern_id"] = DF_proportion_ev_tnbc_baseline["pattern_id"].astype(str)
DF_proportion_ev_tnbc_baseline["dataset"] = "External validation set-2 (Pre-treatment)"
Proportions_ev_tnbc_ot = histograms_ev_tnbc_ot / np.sum(histograms_ev_tnbc_ot, axis=1, keepdims=True)
DF_proportion_ev_tnbc_ot = pd.DataFrame(Proportions_ev_tnbc_ot)
DF_proportion_ev_tnbc_ot = DF_proportion_ev_tnbc_ot.melt(var_name="pattern_id", value_name="proportion")
DF_proportion_ev_tnbc_ot["pattern_id"] = DF_proportion_ev_tnbc_ot["pattern_id"].astype(str)
DF_proportion_ev_tnbc_ot["dataset"] = "External validation set-2 (On-treatment)"
DF_proportion_all = pd.concat([DF_proportion, DF_proportion_iv, DF_proportion_ev, DF_proportion_ev_tnbc_baseline, DF_proportion_ev_tnbc_ot])
for i in range(len(patient_subgroups_discovery)):
    subgroup_id = patient_subgroups_discovery[i]["subgroup_id"]
    characteristic_patterns = patient_subgroups_discovery[i]["characteristic_patterns"]
    for pattern_index in range(len(characteristic_patterns)):
        pattern_id = characteristic_patterns[pattern_index]
        print(subgroup_id, tme_pattern_orders.tolist().index(pattern_id)+1)
f, ax = plt.subplots(11, 6, figsize=(20, 20), sharex=True, tight_layout=True, sharey=True)
for i in range(66):
    sns.stripplot(
        data=DF_proportion_all.loc[DF_proportion_all.pattern_id == str(tme_pattern_orders[i])],
        x = 'dataset',
        y = "proportion",
        palette=sns.color_palette("Set2"),
        ax=ax[i//6, i%6],
    )
    ax[i//6, i%6].spines['top'].set_visible(False)
    ax[i//6, i%6].spines['right'].set_visible(False)
    ax[i//6, i%6].set_title(f"Pattern {i+1}", fontsize=14)
    ax[i//6, i%6].set(xlabel = '', ylabel = '')
    ax[i//6, i%6].set_xlabel('')
    ax[i//6, i%6].get_xaxis().set_visible(False)
plt.show()
f.savefig("Results/figA7.png", dpi=300, bbox_inches='tight')