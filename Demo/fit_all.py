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
from lifelines.statistics import multivariate_logrank_test, logrank_test, pairwise_logrank_test
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
    get_paired_markers
)
sys.path.append("./..")
from cell_graph import Cell_Graph
from bi_graph import BiGraph
from population_graph import Population_Graph
from explainer import Explainer

SC_d_raw = pd.read_csv(os.path.join(PROJECT_ROOT, 'Datasets/Danenberg_et_al/cells.csv'))
survival_d_raw = pd.read_csv(os.path.join(PROJECT_ROOT,'Datasets/Danenberg_et_al/clinical.csv'))
SC_d, SC_iv, survival_d, survival_iv = preprocess_Danenberg(SC_d_raw, survival_d_raw)

bigraph_ = BiGraph(k_patient_clustering = 30)
population_graph_discovery, patient_subgroups_discovery = bigraph_.fit_transform(
    SC_d, 
    survival_data = survival_d
)