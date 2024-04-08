# Bi-level Graph Learning Unveils Prognosis-Relevant Breast Tumor Microenvironment Patterns
BiGraph is an unsupervised learning method for multi-scale discovery of tumor microenvironments (TME). It relies on the construction of a bi-level graph model: (i) a cellular graph, which models the intricate tumor microenvironment, and (ii) a population graph that captures inter-patient similarities, given their respective cellular graphs, by means of a soft Weisfeiler-Lehman (WL) subtree kernel.

### How to use it
```
git clone https://github.com/JasmineZhen218/BiGraph4TME.git
cd BiGraph4TME
```
#### Discovery 
Prepare single-cell data (mandatory) and clinical data (optional) of your discovery set
(conduct any data inclusion, cleaning, and normalization if needed). 
```
import pandas as pd
singleCell_data = pd.read_csv("path_to_single_cell_data.csv") # mandaroty
# each row in `singleCell_data` represents a single cell, and it should at least include the following columns:
    # - `patientID`: patient id; type: string or integer
    # - `imageID`: image id, if each patient has only one image, this it is same as `patientID`; type: string or integer
    # - `celltypeID`: cell type; type: integer
    # - `coorX`: x coordinate of the cell's spatial location; type: float
    # - `coorY`: y coordinate of the cell's spatial location; type: float
survival_data = pd.read_csv(path_to_single_survival data.csv") # optional
# each row in `survival_data` represents a single patient, and it should at least include the following columns:
    # - `patientID`: patient id; type: string or integer
    # - `status`: Survival Status; type: integer. 
        for overall survival: 0: alive; 1: death
        for disease-specific survival: 0: alive; 1: disease-specific death
        for recurrence-free survival: 0: not recurrent; 1: recurrent
    # -- `length`: survival time in month; type: float
```
##### Fit BiGraph model
```
from bi_graph import BiGraph
bigraph_ = BiGraph(
        a=0.01,
        n_iter=0,
        k_subtree_clustering=100,
        k_patient_clustering=30,
        resolution=1.0,
        size_smallest_cluster=10,
        threshold_hodges_lehmann=0.5,
        seed=1,
    )
population_graph, patient_subgroups = bigraph_.fit_transform(
    singleCell_data,
        patientID_colname="patientID",
        imageID_colname="imageID",
        celltypeID_colname="celltypeID",
        coorX_colname="coorX",
        coorY_colname="coorY",
        survival_data=None,
        status_colname="status",
        time_colname="time",
)
##### Discovery results
```
Retrieve signature of identified TME patterns
```
Signatures = bigrah.fitted_soft_wl_subtree_.Signatures
# the shape of Signature is `n_patterns x n_cell_types`
```
Visualize population graph
```
from utils import visualize_population_graph
fig = visualize_population_graph(population_graph)
```
Survival analysis of identified patient subgroups (only applicable when survival_data is available)
```
from utils import plot_survival_curves
fig = plot_survival_curves(patient_subgroups)
```
Retrieve characterisitic patterns for each patient subgroup
```

```
### Reproduce results published in Bi-level Graph Learning Unveils Prognosis-Relevant Breast Tumor Microenvironment Patterns
run Demo/BiGraph Applied to Breast Cancer.ipynb

