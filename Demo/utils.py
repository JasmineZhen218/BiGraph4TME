import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from definitions import patient_ids_discovery, patient_ids_inner_validation, get_node_id

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
from sklearn.neighbors import NearestNeighbors




def preprocess_Danenberg(singleCell_data, survival_data):
    print("Initially,")
    print(
        "{} patients ({} images) with cell data, {} patients with clinical data, ".format(
            len(singleCell_data["metabric_id"].unique()),
            len(singleCell_data["ImageNumber"].unique()),
            len(survival_data["metabric_id"].unique()),
        )
    )
    print("\nRemove images without invasive tumor,")
    singleCell_data = singleCell_data.loc[singleCell_data.isTumour == 1]
    print(
        "{} patients ({} images) with cell data, {} patients with clinical data, ".format(
            len(singleCell_data["metabric_id"].unique()),
            len(singleCell_data["ImageNumber"].unique()),
            len(survival_data["metabric_id"].unique()),
        )
    )
    print("\nRemove patients with no clinical data,")
    singleCell_data = singleCell_data.loc[
        singleCell_data["metabric_id"].isin(survival_data["metabric_id"])
    ]
    print(
        "{} patients ({} images) with cell data and clinical data, ".format(
            len(singleCell_data["metabric_id"].unique()),
            len(singleCell_data["ImageNumber"].unique()),
        )
    )
    # remove images with less than 500 cells
    print("\nRemove images with less than 500 cells")
    cells_per_image = singleCell_data.groupby("ImageNumber").size()
    singleCell_data = singleCell_data.loc[
        singleCell_data["ImageNumber"].isin(
            cells_per_image[cells_per_image > 500].index
        )
    ]
    survival_data = survival_data.loc[
        survival_data["metabric_id"].isin(singleCell_data["metabric_id"].unique())
    ]
    print(
        "{} patients ({} images) with more than 500 cells and clinical data, ".format(
            len(singleCell_data["metabric_id"].unique()),
            len(singleCell_data["ImageNumber"].unique()),
        )
    )
    patient_ids_Danenberg = set(singleCell_data["metabric_id"].unique())
    # check discovery set and inner validation set are disjoint
    assert set(patient_ids_discovery).isdisjoint(set(patient_ids_inner_validation))
    # check discovery set and inner validation set are exhaustive
    assert (
        set(patient_ids_discovery).union(set(patient_ids_inner_validation))
        == patient_ids_Danenberg
    )

    SC_dc = pd.concat(
        [
            singleCell_data.loc[singleCell_data["metabric_id"] == i]
            for i in patient_ids_discovery
        ]
    )
    SC_iv = pd.concat(
        [
            singleCell_data.loc[singleCell_data["metabric_id"] == i]
            for i in patient_ids_inner_validation
        ]
    )
    print("\nAfter splitting into discovery and validation sets,")
    print(
        "{} patients and {} images in the discovery set".format(
            len(patient_ids_discovery),
            len(SC_dc["ImageNumber"].unique()),
        )
    )
    print(
        "{} patients and {} images in the validation set".format(
            len(patient_ids_inner_validation),
            len(SC_iv["ImageNumber"].unique()),
        )
    )
    SC_dc["celltypeID"] = SC_dc["meta_description"].map(
        get_node_id("Danenberg", "CellType")
    )
    SC_dc = SC_dc.rename(
        columns={
            "metabric_id": "patientID",
            "ImageNumber": "imageID",
            "Location_Center_X": "coorX",
            "Location_Center_Y": "coorY",
        }
    )
    SC_iv["celltypeID"] = SC_iv["meta_description"].map(
        get_node_id("Danenberg", "CellType")
    )
    SC_iv = SC_iv.rename(
        columns={
            "metabric_id": "patientID",
            "ImageNumber": "imageID",
            "Location_Center_X": "coorX",
            "Location_Center_Y": "coorY",
        }
    )
    survival_data = survival_data.rename(
        columns={
            "metabric_id": "patientID",
            "Disease-specific Survival (Months)": "time",
            "Disease-specific Survival Status": "status",
        }
    )
    survival_dc = survival_data.loc[
        survival_data["patientID"].isin(patient_ids_discovery)
    ]
    survival_dc["status"] = survival_dc["status"].map({"0:LIVING": 0, "1:DECEASED": 1})
    survival_iv = survival_data.loc[
        survival_data["patientID"].isin(patient_ids_inner_validation)
    ]
    survival_iv["status"] = survival_iv["status"].map({"0:LIVING": 0, "1:DECEASED": 1})
    return SC_dc, SC_iv, survival_dc, survival_iv


def preprocess_Jackson(singleCell_data, survival_data):
    print("Initially,")
    print(
        "{} patients, {} images, and {} cells".format(
            len(survival_data["PID"].unique()),
            len(survival_data["core"].unique()),
            len(singleCell_data),
        )
    )
    print("\nRemove normal breast samples")
    survival_data = survival_data[survival_data["diseasestatus"] == "tumor"]
    print(
        "{} patients, {} images, and {} cells".format(
            len(survival_data["PID"].unique()),
            len(survival_data["core"].unique()),
            len(singleCell_data),
        )
    )
    # remove images with less than 500 cells
    print("\nRemove images with less than 500 cells")
    cells_per_image = singleCell_data.groupby("core").size()
    singleCell_data = singleCell_data.loc[
        singleCell_data["core"].isin(cells_per_image[cells_per_image > 500].index)
    ]
    singleCell_data = pd.merge(singleCell_data, survival_data, on="core", how="inner")
    print(
        "{} patients, {} images, and {} cells".format(
            len(singleCell_data["PID"].unique()),
            len(singleCell_data["core"].unique()),
            len(singleCell_data),
        )
    )
    SC_ev = singleCell_data
    survival_ev = survival_data

    SC_ev["celltypeID"] = SC_ev["cell_type"].map(get_node_id("Jackson", "CellType"))
    SC_ev = SC_ev.rename(
        columns={
            "PID": "patientID",
            "core": "imageID",
            "Location_Center_X": "coorX",
            "Location_Center_Y": "coorY",
        }
    )

    survival_ev = survival_ev.rename(
        columns={
            "PID": "patientID",
            "OSmonth": "time",
            "Patientstatus": "status",
        }
    )
    survival_ev["status"] = survival_ev["status"].map(
        {
            "death by primary disease": 1,
            "alive w metastases": 0,
            "death": 1,
            "alive": 0,
        }
    )

    return SC_ev, survival_ev

def preprocess_Wang(singleCell_data, survival_data):
    print("Initially,")
    print(
        "{} patients, {} images, and {} cells".format(
            len(singleCell_data["PatientID"].unique()),
            len(singleCell_data["ImageID"].unique()),
            len(singleCell_data),
        )
    )
    # print("\Only keep Pre-treatment samples")
    # survival_data = survival_data[survival_data["BiopsyPhase"] == "Baseline"]
    # singleCell_data = singleCell_data[singleCell_data['BiopsyPhase'] == 'Baseline']
    # print(
    #     "{} patients, {} images, and {} cells".format(
    #         len(singleCell_data["PatientID"].unique()),
    #         len(singleCell_data["ImageID"].unique()),
    #         len(singleCell_data),
    #     )
    # )
    # remove images with less than 500 cells
    print("\nRemove images with less than 500 cells")
    cells_per_image = singleCell_data.groupby("ImageID").size()
    singleCell_data = singleCell_data.loc[
        singleCell_data["ImageID"].isin(cells_per_image[cells_per_image > 500].index)
    ]
    print(
        "{} patients, {} images, and {} cells".format(
            len(singleCell_data["PatientID"].unique()),
            len(singleCell_data["ImageID"].unique()),
            len(singleCell_data),
        )
    )
    SC_ev = singleCell_data
    survival_ev = survival_data

    SC_ev["celltypeID"] = SC_ev["Label"].map(get_node_id("Wang", "CellType"))
    SC_ev = SC_ev.rename(
        columns={
            "PatientID": "patientID",
            "ImageNumber": "imageID",
            "Location_Center_X": "coorX",
            "Location_Center_Y": "coorY",
        }
    )

    survival_ev = survival_ev.rename(
        columns={
            "PatientID": "patientID"}
    )


    return SC_ev, survival_ev



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
