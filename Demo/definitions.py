import seaborn as sns
def get_node_id(dataset, node_label):
    if dataset == "Danenberg":
        if node_label == "CellType":
            cell_phenotype_id = {
                "CK^{med}ER^{lo}": 0,
                "ER^{hi}CXCL12^{+}": 1,
                "CK^{lo}ER^{lo}": 2,
                "Ep Ki67^{+}": 3,
                "CK^{lo}ER^{med}": 4,
                "Ep CD57^{+}": 5,
                "MHC I & II^{hi}": 6,
                "Basal": 7,
                "HER2^{+}": 8,
                "CK8-18^{hi}CXCL12^{hi}": 9,
                "CK^{+} CXCL12^{+}": 10,
                "CK8-18^{hi}ER^{lo}": 11,
                "CK8-18^{+} ER^{hi}": 12,
                "MHC I^{hi}CD57^{+}": 13,
                "MHC^{hi}CD15^{+}": 14,
                "CD15^{+}": 15,
                "CD4^{+} T cells & APCs": 16,
                "CD4^{+} T cells": 17,
                "CD8^{+} T cells": 18,
                "T_{Reg} & T_{Ex}": 19,
                "B cells": 20,
                "CD38^{+} lymphocytes": 21,
                "Macrophages": 22,
                "Granulocytes": 23,
                "Macrophages & granulocytes": 24,
                "CD57^{+}": 25,
                "Ki67^{+}": 26,
                "Endothelial": 27,
                "Fibroblasts": 28,
                "Fibroblasts FSP1^{+}": 29,
                "Myofibroblasts PDPN^{+}": 30,
                "Myofibroblasts": 31,
            }
        elif node_label == "CellCategory":
            cell_phenotype_id = {
                "tumor": 0,
                "immune": 1,
                "stromal": 2,
            }
        elif node_label == "TMECellType":
            cell_phenotype_id = {
                "tumor": 0,
                "CD4^{+} T cells & APCs": 1,
                "CD4^{+} T cells": 2,
                "CD8^{+} T cells": 3,
                "T_{Reg} & T_{Ex}": 4,
                "B cells": 5,
                "CD38^{+} lymphocytes": 6,
                "Macrophages": 7,
                "Granulocytes": 8,
                "Macrophages & granulocytes": 9,
                "CD57^{+}": 10,
                "Ki67^{+}": 11,
                "Endothelial": 12,
                "Fibroblasts": 13,
                "Fibroblasts FSP1^{+}": 14,
                "Myofibroblasts PDPN^{+}": 15,
                "Myofibroblasts": 16,
            }
    elif dataset == "Jackson":
        cell_phenotype_id = {
            "B cells": 0,
            "B and T cells": 1,
            "T cells_1": 2,
            "Macrophages_1": 3,
            "T cells_2": 4,
            "Macrophages_2": 5,
            "Endothelial": 6,
            "Vimentin hi": 7,
            "small circular": 8,
            "small elongated": 9,
            "Fibronectin hi": 10,
            "larger elongated": 11,
            "SMA hi Vimentin": 12,
            "hypoxic": 13,
            "apoptotic": 14,
            "proliferative": 15,
            "p53 EGFR": 16,
            "Basal CK": 17,
            "CK7 CK hi Cadherin": 18,
            "CK7 CK": 19,
            "Epithelial low": 20,
            "CK low HR low": 21,
            "HR hi CK": 22,
            "CK HR": 23,
            "HR low CK": 24,
            "CK low HR hi p53": 25,
            "Myoepithalial": 26,
        }
    return cell_phenotype_id


def get_node_color(dataset, node_label):
    if dataset == "Danenberg":
        if node_label == "CellType":
            cell_color = {
                "CK^{med}ER^{lo}": "#40647A",
                "ER^{hi}CXCL12^{+}": "#99CCCC",
                "CD4^{+} T cells & APCs": "#F8B195",
                "CD4^{+} T cells": "#FF50A2",
                "Endothelial": "#FFC400",
                "Fibroblasts": "#007B1D",
                "Myofibroblasts PDPN^{+}": "#c6ce2b",
                "CD8^{+} T cells": "#D6316F",
                "CK8-18^{hi}CXCL12^{hi}": "#00BFFF",
                "Myofibroblasts": "#81ca33",
                "CK^{lo}ER^{lo}": "#73A9DF",
                "Macrophages": "#800080",
                "CK^{+} CXCL12^{+}": "#1E90FF",
                "CK8-18^{hi}ER^{lo}": "#006BD7",
                "CK8-18^{+} ER^{hi}": "#005A9C",
                "CD15^{+}": "#D4CFC9",
                "MHC I & II^{hi}": "#8F756B",
                "T_{Reg} & T_{Ex}": "#FE7A15",
                "CD57^{+}": "#D699B4",
                "Ep Ki67^{+}": "#000066",
                "CK^{lo}ER^{med}": "#0088B2",
                "Macrophages & granulocytes": "#d6bbf6",
                "CD38^{+} lymphocytes": "#FBFEC9",
                "Ki67^{+}": "#006400",
                "HER2^{+}": "#D3E7EE",
                "B cells": "#FFFF00",
                "Basal": "#805B3B",
                "Fibroblasts FSP1^{+}": "#37AD3F",
                "Granulocytes": "#6C5B7B",
                "MHC I^{hi}CD57^{+}": "#A69287",
                "Ep CD57^{+}": "#708090",
                "MHC^{hi}CD15^{+}": "#F4A460",
            }
        elif node_label == "CellCategory":
            cell_color = {
                "tumor": "blue",
                "immune": "red",
                "stroma": "green",
            }
        elif node_label == "TMECellType":
            cell_color = {
                "tumor": "blue",
                "CD4^{+} T cells & APCs": "#F8B195",
                "CD4^{+} T cells": "#FF50A2",
                "Endothelial": "#FFC400",
                "Fibroblasts": "#007B1D",
                "Myofibroblasts PDPN^{+}": "#c6ce2b",
                "CD8^{+} T cells": "#D6316F",
                "Myofibroblasts": "#81ca33",
                "Macrophages": "#800080",
                "CD15^{+}": "#D4CFC9",
                "T_{Reg} & T_{Ex}": "#FE7A15",
                "CD57^{+}": "#D699B4",
                "Macrophages & granulocytes": "#d6bbf6",
                "CD38^{+} lymphocytes": "#FBFEC9",
                "Ki67^{+}": "#006400",
                "B cells": "#FFFF00",
                "Fibroblasts FSP1^{+}": "#37AD3F",
                "Granulocytes": "#6C5B7B",
            }
    return cell_color


def get_paired_markers(source="Danenberg", target="Jackson"):
    if (source == "Danenberg") and (target == "Jackson"):
        Paired_Markers = [
            ("panCK", "AE1/AE3"),
            ("CK8-18", "CK8/18"),
            ("CK5", "CK5"),
            ("ER", "ER"),
            ("HER2 (3B5)", "Her2"),
            ("HER2 (D8F12)", "Her2"),
            ("CD31-vWF", "vWF/CD31"),
            ("SMA", "SMA"),
            ("Ki-67", "Ki-67"),
            ("c-Caspase3", "cleaved PARP/cleaved Caspase3"),
            ("CD45RA", "CD45"),
            ("CD3", "CD3"),
            ("CD20", "CD20"),
            ("CD68", "CD68"),
            ("Histone H3", "Histone H3_1"),
            ("Histone H3", "Histone H3_2"),
            ("DNA1", "DNA1"),
        ]
    return Paired_Markers


Protein_list_Danenberg = [
    "panCK",
    "CK8-18",
    "CK5",
    "ER",
    "HER2 (3B5)",
    "CD31-vWF",
    "SMA",
    "CD45RO",
    "CD3",
    "CD20",
    "CD68",
    "Ki-67",
    "c-Caspase3",
    "Histone H3",
    "DNA1",
    "CD15",
    "CD57",
    "CXCL12",
    "FSP1",
    "Caveolin-1",
    "Podoplanin",
    "PDGFRB",
    "HLA-ABC",
    "CD4",
    "CD8",
    "CD38",
    "FOXP3",
    "CD16",
    "ICOS",
    "CD163",
    "CD11c",
    "PD-1",
    "OX40",
    "GITR",
    "HLA-DR",
    "B2M",
]
Protein_list_Jackson = [
    "AE1/AE3",
    "CK8/18",
    "CK5",
    "ER",
    "Her2",
    "vWF/CD31",
    "SMA",
    "CD45",
    "CD3",
    "CD20",
    "CD68",
    "Ki-67",
    "cleaved PARP/cleaved Caspase3",
    "Histone H3_1",
    "DNA1",
    "CK7",
    "CK14",
    "CK19",  # # Cytokeratins(CK) 7
    "E/P-Cadherin",
    "CD44",  # Adhesion Molecules
    "PR",  # Hormone Receprtors
    "EGFR",  # RTK Signaling
    # Endothelial
    "Vimentin",
    "Fibronectin",  # Mesenchymal Markers
    # Immune Context
    "CAIX",  # Hypoxia
    "p53",
    "cMyc",
    "GATA3",
    "Twist",
    "Slug",  # Transcription Factors
    "p-HH3",
    "p-S6",
    "p-mTOR",  # Cell Growth and Division
    # Cell Death
]


Protein_list_display_Danenberg = [
    "panCK",
    "CK8-18",
    "CK5",
    "ER",
    "HER2",
    "CD31-vWF",
    "SMA",
    "CD45",
    "CD3",
    "CD20",
    "CD68",
    "Ki-67",
    "c-Caspase3",
    "Histone H3",
    "DNA",
    "CD15",
    "CD57",
    "CXCL12",
    "FSP1",
    "Caveolin-1",
    "Podoplanin",
    "PDGFRB",
    "HLA-ABC",
    "CD4",
    "CD8",
    "CD38",
    "FOXP3",
    "CD16",
    "ICOS",
    "CD163",
    "CD11c",
    "PD-1",
    "OX40",
    "GITR",
    "HLA-DR",
    "B2M",
]
Protein_list_display_Jackson = [
    "PanCK",
    "CK8/18",
    "CK5",
    "ER",
    "Her2",
    "vWF/CD31",
    "SMA",
    "CD45",
    "CD3",
    "CD20",
    "CD68",
    "Ki-67",
    "c-Caspase3",
    "Histone H3_1",
    "DNA1",
    "CK7",
    "CK14",
    "CK19",  # # Cytokeratins(CK) 7
    "E/P-Cadherin",
    "CD44",  # Adhesion Molecules
    "PR",  # Hormone Receprtors
    "EGFR",  # RTK Signaling
    # Endothelial
    "Vimentin",
    "Fibronectin",  # Mesenchymal Markers
    # Immune Context
    "CAIX",  # Hypoxia
    "p53",
    "cMyc",
    "GATA3",
    "Twist",
    "Slug",  # Transcription Factors
    "p-HH3",
    "p-S6",
    "p-mTOR",  # Cell Growth and Division
]

color_palette_clinical = {
    "Unknown": "grey",
    'HR+/HER2+': sns.color_palette("Set3")[0],
    'HR-/HER2+': sns.color_palette("Set3")[4],
    'HER2+':sns.color_palette("tab10")[9],
    'TNBC': sns.color_palette("Set3")[5],
    'HR+/HER2-': sns.color_palette("Set3")[7],
}
color_palette_Bigraph = {
    "Unclassified":"white",
    'S1': sns.color_palette("tab10")[0],
    'S2': sns.color_palette("tab10")[1],
    'S3': sns.color_palette("tab10")[2],
    'S4': sns.color_palette("tab10")[3],
    'S5': sns.color_palette("tab10")[4],
    'S6': sns.color_palette("tab10")[5],
    'S7': sns.color_palette("tab10")[6],
}


Cell_types_displayed_Danenberg = [
    r"CK$^{med}$ER$^{lo}$",
    r"ER$^{hi}$CXCL12$^{+}$",
    r"CK$^{lo}$ER$^{lo}$",
    r"Ep Ki67$^{+}$",
    r"CK$^{lo}$ER$^{med}$",
    r"Ep CD57$^{+}$",
    r"MHC I & II$^{hi}$",
    "Basal",
    r"HER2$^{+}$",
    r"CK8-18$^{hi}$CXCL12$^{hi}$",
    r"CK$^{+}$ CXCL12$^{+}$",
    r"CK8-18$^{hi}$ER$^{lo}$",
    r"CK8-18$^{+}$ ER$^{hi}$",
    r"MHC I$^{hi}$CD57$^{+}$",
    r"MHC$^{hi}$CD15$^{+}$",
    r"CD15$^{+}$",
    r"CD4$^{+}$ T cells & APCs",
    r"CD4$^{+}$ T cells",
    r"CD8$^{+}$ T cells",
    r"T$_{Reg}$ & T$_{Ex}$",
    "B cells",
    r"CD38$^{+}$ lymphocytes",
    "Macrophages",
    "Granulocytes",
    "Macrophages & granulocytes",
    r"CD57$^{+}$",
    r"Ki67$^{+}$",
    "Endothelial",
    "Fibroblasts",
    r"Fibroblasts FSP1$^{+}$",
    r"Myofibroblasts PDPN$^{+}$",
    "Myofibroblasts",
]


patient_ids_discovery = [
    "MB-0099",
    "MB-0598",
    "MB-0606",
    "MB-0247",
    "MB-0528",
    "MB-0091",
    "MB-0211",
    "MB-0442",
    "MB-0321",
    "MB-0343",
    "MB-0569",
    "MB-0556",
    "MB-0557",
    "MB-0236",
    "MB-0317",
    "MB-0351",
    "MB-0353",
    "MB-0354",
    "MB-0356",
    "MB-0357",
    "MB-0869",
    "MB-0874",
    "MB-0877",
    "MB-0350",
    "MB-0358",
    "MB-0882",
    "MB-0893",
    "MB-0904",
    "MB-0906",
    "MB-0360",
    "MB-0891",
    "MB-0272",
    "MB-0319",
    "MB-0045",
    "MB-0081",
    "MB-0114",
    "MB-0316",
    "MB-0897",
    "MB-0899",
    "MB-0537",
    "MB-0193",
    "MB-0220",
    "MB-0445",
    "MB-0226",
    "MB-0229",
    "MB-0305",
    "MB-0307",
    "MB-0542",
    "MB-0301",
    "MB-0302",
    "MB-0543",
    "MB-0324",
    "MB-0325",
    "MB-0309",
    "MB-0532",
    "MB-0255",
    "MB-0256",
    "MB-0258",
    "MB-0529",
    "MB-0530",
    "MB-0269",
    "MB-0333",
    "MB-0260",
    "MB-0262",
    "MB-0264",
    "MB-0352",
    "MB-0056",
    "MB-0066",
    "MB-0036",
    "MB-0068",
    "MB-0101",
    "MB-0102",
    "MB-0083",
    "MB-0216",
    "MB-0590",
    "MB-0591",
    "MB-0366",
    "MB-0588",
    "MB-0599",
    "MB-0620",
    "MB-0621",
    "MB-0600",
    "MB-0603",
    "MB-0662",
    "MB-0607",
    "MB-0290",
    "MB-0279",
    "MB-0914",
    "MB-0283",
    "MB-0285",
    "MB-0613",
    "MB-0614",
    "MB-0616",
    "MB-0287",
    "MB-0568",
    "MB-0300",
    "MB-0292",
    "MB-0245",
    "MB-0010",
    "MB-0013",
    "MB-0111",
    "MB-0112",
    "MB-0014",
    "MB-0018",
    "MB-0028",
    "MB-0132",
    "MB-0140",
    "MB-0142",
    "MB-0143",
    "MB-0134",
    "MB-0135",
    "MB-0136",
    "MB-0163",
    "MB-0167",
    "MB-0153",
    "MB-0176",
    "MB-0177",
    "MB-0191",
    "MB-0118",
    "MB-0185",
    "MB-0116",
    "MB-0188",
    "MB-0123",
    "MB-0195",
    "MB-0107",
    "MB-0200",
    "MB-0202",
    "MB-0368",
    "MB-0378",
    "MB-0384",
    "MB-0385",
    "MB-0372",
    "MB-0397",
    "MB-0401",
    "MB-0390",
    "MB-0393",
    "MB-0395",
    "MB-0641",
    "MB-0642",
    "MB-0644",
    "MB-0646",
    "MB-0660",
    "MB-0661",
    "MB-0227",
    "MB-0653",
    "MB-0659",
    "MB-0165",
    "MB-0174",
    "MB-0389",
    "MB-0443",
    "MB-0455",
    "MB-0461",
    "MB-0469",
    "MB-0487",
    "MB-0394",
    "MB-0410",
    "MB-0419",
    "MB-0426",
    "MB-0414",
    "MB-0448",
    "MB-0449",
    "MB-0434",
    "MB-0437",
    "MB-0463",
    "MB-0465",
    "MB-0454",
    "MB-0457",
    "MB-0459",
    "MB-0474",
    "MB-0486",
    "MB-0488",
    "MB-0491",
    "MB-0475",
    "MB-0484",
    "MB-0485",
    "MB-0513",
    "MB-0518",
    "MB-0498",
    "MB-0501",
    "MB-0519",
    "MB-0574",
    "MB-0582",
    "MB-0583",
    "MB-0626",
    "MB-0575",
    "MB-0576",
    "MB-0578",
    "MB-0579",
    "MB-0580",
    "MB-0563",
    "MB-0339",
    "MB-0340",
    "MB-0145",
    "MB-0207",
    "MB-0314",
    "MB-0336",
    "MB-0562",
    "MB-0035",
    "MB-0312",
    "MB-0525",
    "MB-0526",
    "MB-0235",
    "MB-0238",
    "MB-0040",
    "MB-0064",
    "MB-0119",
    "MB-0470",
    "MB-0596",
    "MB-0610",
    "MB-0520",
    "MB-0128",
    "MB-0347",
    "MB-0348",
    "MB-0381",
    "MB-0921",
    "MB-2969",
    "MB-2513",
    "MB-2616",
    "MB-2629",
    "MB-2536",
    "MB-2556",
    "MB-2578",
    "MB-2610",
    "MB-2614",
    "MB-3002",
    "MB-2641",
    "MB-2721",
    "MB-2752",
    "MB-2730",
    "MB-2742",
    "MB-2760",
    "MB-2769",
    "MB-2770",
    "MB-2771",
    "MB-2772",
    "MB-2792",
    "MB-2796",
    "MB-2844",
    "MB-2849",
    "MB-2838",
    "MB-2843",
    "MB-2912",
    "MB-2854",
    "MB-2857",
    "MB-2947",
    "MB-2951",
    "MB-2954",
    "MB-2929",
    "MB-2931",
    "MB-3560",
    "MB-3692",
    "MB-3978",
    "MB-3567",
    "MB-3576",
    "MB-3600",
    "MB-3747",
    "MB-3752",
    "MB-3082",
    "MB-3035",
    "MB-3121",
    "MB-3165",
    "MB-3087",
    "MB-3167",
    "MB-3253",
    "MB-3218",
    "MB-3315",
    "MB-3329",
    "MB-3292",
    "MB-3300",
    "MB-3350",
    "MB-3370",
    "MB-3381",
    "MB-3361",
    "MB-3363",
    "MB-3386",
    "MB-3430",
    "MB-3436",
    "MB-3389",
    "MB-3403",
    "MB-3487",
    "MB-3492",
    "MB-3495",
    "MB-3453",
    "MB-3459",
    "MB-3466",
    "MB-3510",
    "MB-3525",
    "MB-3526",
    "MB-3528",
    "MB-0398",
    "MB-0148",
    "MB-0412",
    "MB-0175",
    "MB-0328",
    "MB-0152",
    "MB-0551",
    "MB-0460",
    "MB-0545",
    "MB-0573",
    "MB-0151",
    "MB-0571",
    "MB-0404",
    "MB-0201",
    "MB-0481",
    "MB-0650",
    "MB-0349",
    "MB-0105",
    "MB-0221",
    "MB-0280",
    "MB-0311",
    "MB-0240",
    "MB-0254",
    "MB-0252",
    "MB-0534",
    "MB-0059",
    "MB-0043",
    "MB-0100",
    "MB-0103",
    "MB-0106",
    "MB-0006",
    "MB-0079",
    "MB-0593",
    "MB-0622",
    "MB-0605",
    "MB-0288",
    "MB-0030",
    "MB-0130",
    "MB-0022",
    "MB-0139",
    "MB-0166",
    "MB-0120",
    "MB-0208",
    "MB-0382",
    "MB-0383",
    "MB-0371",
    "MB-0388",
    "MB-0387",
    "MB-0629",
    "MB-0154",
    "MB-0168",
    "MB-0077",
    "MB-0408",
    "MB-0422",
    "MB-0416",
    "MB-0418",
    "MB-0441",
    "MB-0464",
    "MB-0458",
    "MB-0479",
    "MB-0515",
    "MB-0624",
    "MB-0246",
    "MB-0910",
    "MB-0411",
    "MB-0075",
    "MB-0505",
    "MB-0451",
    "MB-0367",
    "MB-0405",
    "MB-2993",
    "MB-2977",
    "MB-2669",
    "MB-2728",
    "MB-3006",
    "MB-2810",
    "MB-2848",
    "MB-2828",
    "MB-2896",
    "MB-2900",
    "MB-2863",
    "MB-3702",
    "MB-3667",
    "MB-3823",
    "MB-3781",
    "MB-3088",
    "MB-3089",
    "MB-3360",
    "MB-3435",
    "MB-3388",
]


patient_ids_inner_validation = [
    "MB-0282",
    "MB-0190",
    "MB-0222",
    "MB-0224",
    "MB-0369",
    "MB-0521",
    "MB-0555",
    "MB-0570",
    "MB-0345",
    "MB-0363",
    "MB-0901",
    "MB-0884",
    "MB-0361",
    "MB-0362",
    "MB-0271",
    "MB-0278",
    "MB-0002",
    "MB-0005",
    "MB-0270",
    "MB-0535",
    "MB-0115",
    "MB-0232",
    "MB-0291",
    "MB-0482",
    "MB-0483",
    "MB-0225",
    "MB-0198",
    "MB-0322",
    "MB-0304",
    "MB-0549",
    "MB-0266",
    "MB-0261",
    "MB-0536",
    "MB-0359",
    "MB-0046",
    "MB-0085",
    "MB-0089",
    "MB-0097",
    "MB-0589",
    "MB-0594",
    "MB-0584",
    "MB-0601",
    "MB-0608",
    "MB-0609",
    "MB-0612",
    "MB-0296",
    "MB-0126",
    "MB-0131",
    "MB-0144",
    "MB-0137",
    "MB-0161",
    "MB-0150",
    "MB-0180",
    "MB-0181",
    "MB-0117",
    "MB-0109",
    "MB-0122",
    "MB-0206",
    "MB-0203",
    "MB-0370",
    "MB-0373",
    "MB-0375",
    "MB-0391",
    "MB-0396",
    "MB-0643",
    "MB-0631",
    "MB-0637",
    "MB-0664",
    "MB-0666",
    "MB-0648",
    "MB-0652",
    "MB-0915",
    "MB-0194",
    "MB-0050",
    "MB-0400",
    "MB-0413",
    "MB-0438",
    "MB-0421",
    "MB-0425",
    "MB-0428",
    "MB-0440",
    "MB-0446",
    "MB-0429",
    "MB-0453",
    "MB-0467",
    "MB-0490",
    "MB-0492",
    "MB-0476",
    "MB-0494",
    "MB-0495",
    "MB-0508",
    "MB-0510",
    "MB-0627",
    "MB-0581",
    "MB-0289",
    "MB-0512",
    "MB-0544",
    "MB-0538",
    "MB-0527",
    "MB-0379",
    "MB-0184",
    "MB-0628",
    "MB-0060",
    "MB-0095",
    "MB-2957",
    "MB-2994",
    "MB-2971",
    "MB-2617",
    "MB-2564",
    "MB-2686",
    "MB-2708",
    "MB-2718",
    "MB-2645",
    "MB-2750",
    "MB-2758",
    "MB-2774",
    "MB-2781",
    "MB-2791",
    "MB-2820",
    "MB-2801",
    "MB-2847",
    "MB-2834",
    "MB-2904",
    "MB-2858",
    "MB-2867",
    "MB-2922",
    "MB-2923",
    "MB-3674",
    "MB-3680",
    "MB-3688",
    "MB-3606",
    "MB-3614",
    "MB-3797",
    "MB-3804",
    "MB-3707",
    "MB-3754",
    "MB-3102",
    "MB-3181",
    "MB-3272",
    "MB-3275",
    "MB-3382",
    "MB-3357",
    "MB-3367",
    "MB-3427",
    "MB-3395",
    "MB-3396",
    "MB-3412",
    "MB-3479",
    "MB-3488",
    "MB-3462",
    "MB-3467",
    "MB-3536",
    "MB-3876",
    "MB-0149",
    "MB-0431",
    "MB-0399",
    "MB-0129",
    "MB-0158",
    "MB-0406",
    "MB-0511",
    "MB-0249",
    "MB-0880",
    "MB-0294",
    "MB-0223",
    "MB-0320",
    "MB-0552",
    "MB-0263",
    "MB-0073",
    "MB-0595",
    "MB-0286",
    "MB-0138",
    "MB-0178",
    "MB-0121",
    "MB-0189",
    "MB-0386",
    "MB-0657",
    "MB-0415",
    "MB-0439",
    "MB-0462",
    "MB-0502",
    "MB-0503",
    "MB-0504",
    "MB-0436",
    "MB-0523",
    "MB-0170",
    "MB-0604",
    "MB-0192",
    "MB-0586",
    "MB-0377",
    "MB-2984",
    "MB-3020",
    "MB-2786",
    "MB-3013",
    "MB-2933",
    "MB-3637",
    "MB-3706",
    "MB-3824",
    "MB-3711",
    "MB-3355",
    "MB-3502",
]
