import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def process_Danenberg_clinical_data(DF_clinical):
    DF_clinical.rename(
        columns={
            "Age at Diagnosis": "Age",
            "PR Status": "PRStatus",
            "HER2 Status": "HER2Status",
        },
        inplace=True,
    )
    DF_clinical["ERStatus"] = DF_clinical["ERStatus"].map({"pos": 1, "neg": 0})
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


def calculate_hazard_ratio(length, status, community_id, adjust_dict={}):
    """
    Calculate hazard ratio for each community
    :param length: length of follow-up
    :param status: status of follow-up
    :param community_id: community id
    :return: HR: hazard ratio for each community
    """
    # Calculate hazard ratio
    HR = []
    cph = CoxPHFitter()
    unique_community_id = list(set(community_id))
    for i in unique_community_id:
        community_id = np.array(community_id)
        DF = pd.DataFrame(
            {"length": length, "status": status, "community": community_id == i}
        )
        for key, value in adjust_dict.items():
            DF[key] = value
        cph.fit(
            DF,
            duration_col="length",
            event_col="status",
            show_progress=False,
        )
        HR.append(
            {
                "community_id": i,
                # "status": np.array(status)[community_id == i],
                # "length": np.array(length)[community_id == i],
                "hr": cph.hazard_ratios_["community"],
                "hr_lower": np.exp(
                    cph.confidence_intervals_["95% lower-bound"]["community"]
                ),
                "hr_upper": np.exp(
                    cph.confidence_intervals_["95% upper-bound"]["community"]
                ),
                "p": cph.summary["p"]["community"],
            }
        )
    return HR
