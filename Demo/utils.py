import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def process_Danenberg_clinical_data(DF_clinical):
    DF_clinical.rename(
        columns={
            "Age at Diagnosis": "Age",
            'PR Status': 'PRStatus',
            'HER2 Status': 'HER2Status',
        },
        inplace=True,
    )
    DF_clinical["ERStatus"] = DF_clinical["ERStatus"].map(
        {"pos": 1, "neg": 0}
    )
    DF_clinical["PRStatus"] = DF_clinical["PRStatus"].map(
        {"Positive": 1, "Negative": 0}
    )
    DF_clinical["HER2Status"] = DF_clinical["HER2Status"].map(
        {"Positive": 1, "Negative": 0}
    )

    for index, row in DF_clinical.iterrows():
        er = row["ERStatus"]
        pr = row["PRStatus"]
        her2 = row["HER2Status"]
        # print(er, pr, her2)
        if (er == 1 or pr == 1) and her2 == 0:
            DF_clinical.loc[index, "Clinical_Subtype"] = "HR+/HER2-"
            DF_clinical.loc[index, "HR_p_HER_n"] = 1
            DF_clinical.loc[index, "HR_p_HER_p"] = 0
            DF_clinical.loc[index, "HR_n_HER_n"] = 0
            DF_clinical.loc[index, "TNBC"] = 0
        elif (er == 1 or pr == 1) and her2 == 1:
            DF_clinical.loc[index, "Clinical_Subtype"] = "HR+/HER2+"
            DF_clinical.loc[index, "HR_p_HER_n"] = 0
            DF_clinical.loc[index, "HR_p_HER_p"] = 1
            DF_clinical.loc[index, "HR_n_HER_n"] = 0
            DF_clinical.loc[index, "TNBC"] = 0
        elif er == 0 and pr == 0 and her2 == 1:
            DF_clinical.loc[index, "Clinical_Subtype"] = "HR-/HER2+"
            DF_clinical.loc[index, "HR_p_HER_n"] = 0
            DF_clinical.loc[index, "HR_p_HER_p"] = 0
            DF_clinical.loc[index, "HR_n_HER_n"] = 1
            DF_clinical.loc[index, "TNBC"] = 0
        elif er == 0 and pr == 0 and her2 == 0:
            DF_clinical.loc[index, "Clinical_Subtype"] = "TNBC"
            DF_clinical.loc[index, "HR_p_HER_n"] = 0
            DF_clinical.loc[index, "HR_p_HER_p"] = 0
            DF_clinical.loc[index, "HR_n_HER_n"] = 0
            DF_clinical.loc[index, "TNBC"] = 1
        else:
            DF_clinical.loc[index, "Clinical_Subtype"] = "Unknown"
            DF_clinical.loc[index, "HR+/HER2-"] = 0
            DF_clinical.loc[index, "HR+/HER2+"] = 0
            DF_clinical.loc[index, "HR-/HER2+"] = 0
            DF_clinical.loc[index, "TNBC"] = 0

    return DF_clinical


def process_Jackson_clinical_data(DF_clinical):
    DF_clinical.rename(columns={"grade": "Grade"}, inplace=True)
    DF_clinical.rename(columns={"age": "Age"}, inplace=True)
    DF_clinical["ERStatus"] = DF_clinical["ERStatus"].map(
        {"positive": 1, "negative": 0}
    )
    DF_clinical["HER2Status"] = DF_clinical["HER2Status"].map(
        {"positive": 1, "negative": 0}
    )
    DF_clinical["PRStatus"] = DF_clinical["PRStatus"].map(
        {"positive": 1, "negative": 0}
    )
    DF_clinical["Overall Survival Status"] = DF_clinical["Overall Survival Status"].map(
        {"0:LIVING": 0, "1:DECEASED": 1}
    )
    DF_clinical["Relapse-free Status (Months)"] = DF_clinical[
        "Relapse Free Status (Months)"
    ].map({"0:LIVING": 0, "1:DECEASED": 1})
    for index, row in DF_clinical.iterrows():
        er = row["ERStatus"]
        pr = row["PRStatus"]
        her2 = row["HER2Status"]
        if (er == 1 or pr == 1) and her2 == 0:
            DF_clinical.loc[index, "Clinical Subtype"] = "HR+/HER2-"
            DF_clinical.loc[index, "HR+/HER2-"] = 1
            DF_clinical.loc[index, "HR+/HER2+"] = 0
            DF_clinical.loc[index, "HR-/HER2+"] = 0
            DF_clinical.loc[index, "TNBC"] = 0
        elif (er == 1 or pr == 1) and her2 == 1:
            DF_clinical.loc[index, "Clinical Subtype"] = "HR+/HER2+"
            DF_clinical.loc[index, "HR+/HER2-"] = 0
            DF_clinical.loc[index, "HR+/HER2+"] = 1
            DF_clinical.loc[index, "HR-/HER2+"] = 0
            DF_clinical.loc[index, "TNBC"] = 0
        elif er == 0 and pr == 0 and her2 == 1:
            DF_clinical.loc[index, "Clinical Subtype"] = "HR-/HER2+"
            DF_clinical.loc[index, "HR+/HER2-"] = 0
            DF_clinical.loc[index, "HR+/HER2+"] = 0
            DF_clinical.loc[index, "HR-/HER2+"] = 1
            DF_clinical.loc[index, "TNBC"] = 0
        elif er == 0 and pr == 0 and her2 == 0:
            DF_clinical.loc[index, "Clinical Subtype"] = "TNBC"
            DF_clinical.loc[index, "HR+/HER2-"] = 0
            DF_clinical.loc[index, "HR+/HER2+"] = 0
            DF_clinical.loc[index, "HR-/HER2+"] = 0
            DF_clinical.loc[index, "TNBC"] = 1
    return DF_clinical



def reverse_dict(D):
    return {v: k for k, v in D.items()}