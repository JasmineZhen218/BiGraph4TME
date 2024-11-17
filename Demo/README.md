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
    ```
    cd Demo/Danenberg_et_al

    ```
- External Validation set -1
    ```
    cd Demo/Jackson_et_al

    ```
- External Validation set -2 (optional): Access to this dataset requires a formal application submitted to the owner. Please check https://zenodo.org/records/7990870 Download `NTPublic/data/derived/clinical.csv` and `NTPublic/data/derived/cells.csv` there after your application gets approved and then place them in `Demo/Wang_et_al`.

## 3. Fit BiGraph Model
```

python fit_all.py
```
## 4. Reproduce figures
```
python fig[X].py
```
Note: Run `fig[X]_tnbc.py` needs external validation set -2 exist in `Demo/Wang_et_al`