# BiGraph4TME
This repo is the official implementation for the `BiGraph` method introduced in **Bi-level Graph Learning Unveils Prognosis-Relevant Breast Tumor Microenvironment Patterns**. 

BiGraph is an unsupervised learning method for multi-scale discovery of tumor microenvironments (TME). It relies on the construction of a bi-level graph model: 
    
 (i) a cellular graph, which models the intricate tumor microenvironment, 

(ii) a population graph that captures inter-patient similarities, given their respective cellular graphs, by means of a **soft** Weisfeiler-Lehman (WL) subtree kernel.

# Requirements
# How to use it
```
git clone https://github.com/JasmineZhen218/BiGraph4TME.git
```
## Data Preparation
Load and preprocess single cell data (mandatory) and survival data (optional).
```
import pandas as pd
SC_d = pd.read_csv("path_to_single_cell_data_of_discovery_set.csv") 
SC_v = pd.read_csv("path_to_single_cell_data_of_validation_set.csv")  

# if survival data available
survival_d = pd.read_csv(path_to_survival_data_of_discovery_set.csv") 
survival_v = pd.read_csv("path_to_survival_data_of_validation_set.csv") 

```
Preprocess data by conducting (user-defined) data inclusion, cleaning, normalization, etc. After preprocessing, each row in `SC_d` and `SC_v` represents a single cell, and it should at least include the following columns:
    
*  `patientID`: patient id; type: string or integer
*  `imageID`: image id, if each patient has only one image, this it is same as `patientID`; type: string or integer
*  `celltypeID`: cell type; type: integer
*  `coorX`: x coordinate of the cell's spatial location; type: float
* `coorY`: y coordinate of the cell's spatial location; type: float

If survival data is available, after preprocessing, each row in `survival_d` and `survival_v` represents a single patient, and it should at least include the following columns:

* `patientID`: patient id, which should match `patientID` in singleCell_d; type: string or integer
* `status`: Survival Status; type: integer. 
    * overall survival: 0: alive; 1: death
    * disease-specific survival: 0: alive; 1: disease-specific death
    * recurrence-free survival: 0: not recurrent; 1: recurrent
* `length`: survival time in month; type: float

## Fit BiGraph model with discovery set
```
from bi_graph import BiGraph
bigraph_ = BiGraph()
population_graph, patient_subgroups = bigraph_.fit_transform(
    SC_d,
    survival_data = survival_d
)
```

## Validate BiGraph model with validation set
```
population_graph_v, patient_subgroups_v = bigraph_.transform(
    SC_v,
    survival_data = survival_v
)

```

# Use Soft-WL subtree kernel individually
### Fit Soft WL subtree kernel with discovery set
```
from soft_wl_subtree import Soft_WL_Subtree
soft_wl_subtree_ = Soft_WL_Subtree()
Similarity_matrix = soft_wl_subtree_.fit_transform(
    SC_d
)
```
### Validate Soft WL subtree kernel with validation set
```
Similarity_matrix = soft_wl_subtree_.transform(
    SC_v
)
```


# Reproduce results in Paper
Check `Demo/BiGraph Applied to Breast Cancer.ipynb`
# Citation
```

```

