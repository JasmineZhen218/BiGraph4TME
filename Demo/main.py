import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from utils import preprocess_Danenberg
sys.path.extend(["./..", "./../.."])

# from bi_graph import BiGraph
# from population_graph import Population_Graph
from bi_graph_knn import BiGraph_KNN
from population_graph_KNN import Population_Graph_KNN

import time

def _sec(s):  # pretty seconds
    return f"{s:.2f}s"

if __name__ == "__main__":
    print("start")
    t0 = time.perf_counter()

    # --- Setup ---
    sys.path.append("./..")
    os.makedirs("Results", exist_ok=True)

    # --- Data Load ---
    dl_start = time.perf_counter()
    SC_d_raw = pd.read_csv("Datasets/Danenberg_et_al/cells.csv")
    survival_d_raw = pd.read_csv("Datasets/Danenberg_et_al/clinical.csv")
    SC_d, _, survival_d, _ = preprocess_Danenberg(SC_d_raw, survival_d_raw)
    dl_end = time.perf_counter()
    print(f"[Timing] Data load + preprocessing: {_sec(dl_end - dl_start)}")

    # --- Model Fit ---
    fit_start = time.perf_counter()
    bigraph = BiGraph_KNN(k_patient_clustering=20, sample_frac=0.5, k_cell_knn=20,
                          soft_wl_save_path="Fitted_swl/Danenberg/fitted_soft_wl_subtree_sample_0.5")


    
    pop_graph, patient_subgroups = bigraph.fit_transform(SC_d, survival_data=survival_d)
    fit_end = time.perf_counter()
    print(f"[Timing] BiGraph_KNN.fit_transform (incl. Soft-WL & population graph): {_sec(fit_end - fit_start)}")

    # --- Subgroup Assignment ---
    sub_start = time.perf_counter()
    patient_ids = SC_d["patientID"].unique()
    subgroup_ids = np.full(len(patient_ids), "Unclassified", dtype=object)
    for subgroup in patient_subgroups:
        idx = np.isin(patient_ids, subgroup["patient_ids"])
        subgroup_ids[idx] = subgroup["subgroup_id"]
    sub_end = time.perf_counter()
    print(f"[Timing] Subgroup assignment: {_sec(sub_end - sub_start)}")

    # --- Color Palette ---
    color_palette = {"Unclassified": "white"}
    color_palette.update({f"S{i+1}": sns.color_palette("tab10")[i] for i in range(10)})

    # === Function: Plot Population Graph ===
    def plot_population_graph(pop_graph, pos, colors, filename, white_nodes=False):
        fig = plt.figure(figsize=(5, 5), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        node_xyz = np.array([pos[v] for v in sorted(pop_graph)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in pop_graph.edges()])

        ax.scatter(
            *node_xyz.T,
            s=60,
            c='white' if white_nodes else [colors[i] for i in subgroup_ids],
            edgecolors="black",
            linewidths=1,
            alpha=1,
        )
        edge_alpha = [
            0.2 * pop_graph[u][v]["weight"] if pop_graph[u][v]["weight"] > 0.1 else 0
            for u, v in pop_graph.edges()
        ]
        for i, (u, v) in enumerate(pop_graph.edges()):
            ax.plot(*edge_xyz[i].T, alpha=edge_alpha[i], color="k")

        ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
        ax.set_xlim(node_xyz[:, 0].min(), node_xyz[:, 0].max())
        ax.set_ylim(node_xyz[:, 1].min(), node_xyz[:, 1].max())
        ax.set_zlim(node_xyz[:, 2].min(), node_xyz[:, 2].max())
        fig.savefig(f"Results/{filename}.svg", dpi=600, bbox_inches="tight")
        plt.close()

    # --- Layout + Plot Population Graph ---
    lay_start = time.perf_counter()
    layout_k = 10 / np.sqrt(len(pop_graph)) if len(pop_graph) > 0 else 0.1
    pos = nx.spring_layout(pop_graph, seed=3, k=layout_k, iterations=100, dim=3)
    lay_end = time.perf_counter()
    print(f"[Timing] spring_layout computation: {_sec(lay_end - lay_start)}")

    plot_start = time.perf_counter()
    plot_population_graph(pop_graph, pos, color_palette, "population_graph", white_nodes=False)
    plot_end = time.perf_counter()
    print(f"[Timing] Population graph plotting: {_sec(plot_end - plot_start)}")

    # --- Survival Data Prep ---
    survprep_start = time.perf_counter()
    lengths = np.array([survival_d.loc[survival_d["patientID"] == pid, "time"].values[0] for pid in patient_ids])
    statuses = np.array([survival_d.loc[survival_d["patientID"] == pid, "status"].values[0] for pid in patient_ids])
    survprep_end = time.perf_counter()
    print(f"[Timing] Survival vectors prep: {_sec(survprep_end - survprep_start)}")

    # === Function: Kaplan-Meier Plot ===
    def plot_km(subgroups, filename, highlight=None):
        f, ax = plt.subplots(figsize=(5, 5))
        kmf = KaplanMeierFitter()

        for subgroup in subgroups:
            sid = subgroup["subgroup_id"]
            idx = (subgroup_ids == sid)
            label = f"{sid} (N={idx.sum()})"
            if highlight and sid in highlight:
                label = f"{sid}: {highlight[sid]} (N={idx.sum()})"
            kmf.fit(lengths[idx], statuses[idx], label=label)
            kmf.plot_survival_function(
                ax=ax,
                ci_show=False,
                color=color_palette[sid],
                show_censors=True,
                linewidth=2,
                censor_styles={"ms": 5, "marker": "|"},
            )

        ax.set_xlabel("Time (Month)", fontsize=14)
        ax.set_ylabel("Cumulative Survival", fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=13, ncol=2 if not highlight else 1, loc="lower left")
        sns.despine()
        f.savefig(f"Results/{filename}.svg", dpi=600, bbox_inches="tight")
        plt.close()

    # --- KM + Hazard plots ---
    km_start = time.perf_counter()
    logrank = multivariate_logrank_test(
        lengths[subgroup_ids != "Unclassified"],
        subgroup_ids[subgroup_ids != "Unclassified"],
        statuses[subgroup_ids != "Unclassified"]
    )
    p_val = logrank.p_value
    plot_km(patient_subgroups, "survivial_curves")
    km_end = time.perf_counter()
    print(f"[Timing] KM + log-rank + curve plotting: {_sec(km_end - km_start)}")

    haz_start = time.perf_counter()
    f, ax = plt.subplots(2, 1, figsize=(8, 3), height_ratios=[5, 1], sharex=True)
    f.subplots_adjust(hspace=0)

    ax[0].hlines(1, -1, len(patient_subgroups), color="k", linestyle="--")
    xticklabels, xtickcolors, N = [], [], []

    for i, subgroup in enumerate(patient_subgroups):
        sid, hr, hr_lb, hr_ub, p = (
            subgroup["subgroup_id"],
            subgroup["hr"],
            subgroup["hr_lower"],
            subgroup["hr_upper"],
            subgroup["p"]
        )
        ax[0].plot([i, i], [hr_lb, hr_ub], color=color_palette[sid], linewidth=2)
        ax[0].scatter([i], [hr], color=color_palette[sid], s=120)
        ax[0].scatter([i], [hr_lb], color=color_palette[sid], s=60, marker="_")
        ax[0].scatter([i], [hr_ub], color=color_palette[sid], s=60, marker="_")

        xticklabels.append(sid)
        xtickcolors.append("k" if p < 0.05 else "gray")
        N.append((subgroup_ids == sid).sum())

    ax[0].set_xticks(range(len(patient_subgroups)))
    ax[0].set_xticklabels(xticklabels)
    for tick, color in zip(ax[1].get_xticklabels(), xtickcolors):
        tick.set_color(color)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Patient Subgroups")
    ax[0].set_ylabel("Hazard ratio")

    sns.barplot(
        x="subgroup_id",
        y="N",
        data=pd.DataFrame({"subgroup_id": xticklabels, "N": N}),
        palette=color_palette,
        ax=ax[1]
    )
    ax[1].invert_yaxis()
    ax[1].set_ylabel("Size")
    ax[1].set_xlabel("")
    f.savefig("Results/hazard_ratios.svg", dpi=600, bbox_inches="tight")
    haz_end = time.perf_counter()
    print(f"[Timing] Hazard ratio plotting: {_sec(haz_end - haz_start)}")

    # --- Totals ---
    t1 = time.perf_counter()
    print(f"[Timing] Total runtime: {_sec(t1 - t0)}")
