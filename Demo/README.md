# Codes for reproducing results in ``Bi-level Graph Learning Unveils Prognosis-Relevant Tumor Microenvironment Patterns from Breast Multiplexed Digital Pathology''

## 1. Install (If not done)
```
git clone https://github.com/JasmineZhen218/BiGraph4TME.git
cd BiGraph4TME
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## 2. Download Single Cell Data
-  Discovery set
    1. Download `cells.csv` at https://drive.google.com/file/d/1TaMnyya2Lpa_s0CKPespBzEHXhUxU-ml/view?usp=drive_link
    2. Place it in `Demo/Datasets/Danenberg_et_al`

- External Validation set -1
    1. Download `cells.csv` at https://drive.google.com/file/d/1JpCFIVCNBWGSUVbmwjHQS1JgPOZhV1Ly/view?usp=drive_link
    2. Place it in ``Demo/Datasets/Jackson_et_al``

- External Validation set -2 (optional): 
    1. Get data access approval from https://zenodo.org/records/7990870 
    2. Download `NTPublic/data/derived/clinical.csv` and `NTPublic/data/derived/cells.csv`
    3. Place them in `Demo/Datasets/Wang_et_al`.

## 3. Fit BiGraph Model
```
cd Demo
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
python fit_all.py
```
## 4. Reproduce figures
```
python fig[X].py
```
Note: Run `fig[X]_tnbc.py` needs external validation set -2 exist in `Demo/Datasets/Wang_et_al`