
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import (
    multivariate_logrank_test,
    logrank_test,
    pairwise_logrank_test,
)
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
from utils import PROJECT_ROOT
from utils import reverse_dict, preprocess_Danenberg, preprocess_Jackson
from definitions import (
    get_node_id,
    get_node_color,
    Protein_list_Danenberg,
    Protein_list_Jackson,
    Protein_list_display_Danenberg,
    Protein_list_display_Jackson,
    Cell_types_displayed_Danenberg,
    Cell_types_displayed_Jackson,
    color_palette_Bigraph,
    color_palette_clinical,
    get_paired_markers,
)
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cell_graph import Cell_Graph
from bi_graph import BiGraph
from population_graph import Population_Graph
from explainer import Explainer
from karateclub.dataset import GraphSetReader
if not os.path.exists("/cis/home/zwang/Projects/BiGraph4TME/Demo/cell_graphs_discovery.pkl"):
    print("load data")
    SC_d_raw = pd.read_csv('/cis/home/zwang/Projects/BiGraph4TME/Demo/Datasets/Danenberg_et_al/cells.csv') 
    survival_d_raw = pd.read_csv('/cis/home/zwang/Projects/BiGraph4TME/Demo/Datasets/Danenberg_et_al/clinical.csv')
    SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)
    print("generate cell graph")
    cell_graph_ = Cell_Graph(
                    a=0.01
                )  # initialize Cell_Graph class with parameter a
    Cell_graphs = cell_graph_.generate(SC_d)  
    To_save = []
    print("convert to nx")
    for patient_id, adj, x in tqdm(Cell_graphs):
        G = nx.from_numpy_matrix(adj)
        nx.set_node_attributes(G, dict(enumerate(x)), "cell_type")
        To_save.append((patient_id, G))
    # reader = GraphSetReader("reddit10k")
    # save the cell graphs
    with open("/cis/home/zwang/Projects/BiGraph4TME/Demo/cell_graphs_discovery.pkl", "wb") as f:
        pickle.dump(To_save, f)
    Cell_graphs_nx = [i[1] for i in To_save]
    Patient_ids = [i[0] for i in To_save]
else:
    print("load cell graph")
    with open("/cis/home/zwang/Projects/BiGraph4TME/Demo/cell_graphs_discovery.pkl", "rb") as f:
        To_save = pickle.load(f)
    Cell_graphs_nx = [i[1] for i in To_save]
    Patient_ids = [i[0] for i in To_save]
# graphs = reader.get_graphs()
# y = reader.get_target()
# print("embed")
if not os.path.exists("/cis/home/zwang/Projects/BiGraph4TME/Demo/IGE_embeddings_discovery.pkl"):

    from karateclub import IGE
    model = IGE()
    model.fit(Cell_graphs_nx)
    X = model.get_embedding()
    # save the embeddings
    with open("/cis/home/zwang/Projects/BiGraph4TME/Demo/IGE_embeddings_discovery.pkl", "wb") as f:
        pickle.dump(X, f)
else:
    with open("/cis/home/zwang/Projects/BiGraph4TME/Demo/IGE_embeddings_discovery.pkl", "rb") as f:
        X = pickle.load(f)

# from sklearn.metrics.pairwise import cosine_similarity
# Similarity_matrix = cosine_similarity(X)
# print("Start generating population graph.")
# population_graph_ = Population_Graph(
#             k_clustering=30,
#             resolution=1,
#             size_smallest_cluster=10,
#             seed=0,
#         )  # initialize Population_Graph class with parameters k, resolution, size_smallest_cluster, and seed
# Population_graph = population_graph_.generate(
#             Similarity_matrix, Patient_ids
#         )  # generate population graph
# print("Population graph generated.")
# print("Start detecting patient subgroups.")
# Patient_subgroups = population_graph_.community_detection(
#             Population_graph
#         )  # detect patient subgroups
# print("Patient subgroups detected.")
# Num_patients_in_subgroups = [
#             len(subgroup["patient_ids"]) for subgroup in Patient_subgroups
#         ]
# print(
#             "There are {} patient subgroups, {} ungrouped patients".format(
#                 len(Patient_subgroups),
#                 len(Patient_ids) - sum(Num_patients_in_subgroups),
#             )
#         )