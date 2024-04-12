from pathlib import Path

import maples_dr
import numpy as np
import pandas as pd
from jppype.utilities.geometric import Point
from maples_dr.dataset import BiomarkerField as Bio
from skimage.measure import label, regionprops


def centroid(map):
    props = regionprops(map.astype(np.uint8))
    return Point(*props[0].centroid)


def connected_components_f1(map1, map2):
    labels1 = label(map1)
    labels2 = label(map2)

    nb_l1 = np.max(labels1)
    nb_l2 = np.max(labels2)

    sen = 0.0
    for l1 in range(1, nb_l1 + 1):
        if np.any((labels1 == l1) & (labels2 > 0)):
            sen += 1
    sen /= nb_l1

    spe = 0.0
    for l2 in range(1, nb_l2 + 1):
        if np.any((labels2 == l2) & (labels1 > 0)):
            spe += 1
    spe /= nb_l2

    f1 = 2 * sen * spe / (sen + spe)
    return f1


def load_annotations_from_scratch():
    FROM_SCRATCH = Path("/home/gaby/These/Data/Fundus/MESSIDOR1500/FromScratch/")
    FROM_SCRACH_RETINOLOGISTS = [
        FROM_SCRATCH / "Marie-Carole Boucher",
        FROM_SCRATCH / "Micheal Brent",
        FROM_SCRATCH / "Renaud Duval",
    ]

    images = [_.stem for _ in (FROM_SCRACH_RETINOLOGISTS[0] / "Microaneurysms").rglob("*.png")]

    fromscratch_datasets = [maples_dr.quick_api.GLOBAL_LOADER.load_dataset(images) for _ in range(3)]

    for r in range(3):
        dataset = fromscratch_datasets[r]
        for lesion, folder in {
            Bio.MICROANEURYSMS: "Microaneurysms",
            Bio.HEMORRHAGES: "Hemorrhages",
            Bio.NEOVASCULARIZATION: "Neovascularization",
            Bio.RED_UNCERTAINS: "Uncertain - Red",
            Bio.EXUDATES: "Exudates",
            Bio.COTTON_WOOL_SPOTS: "Cotton Wool Spots",
            Bio.DRUSENS: "Drusen",
            Bio.BRIGHT_UNCERTAINS: "Uncertain - Bright",
            Bio.MACULA: "Macula",
            Bio.OPTIC_DISC: "Disk",
            Bio.OPTIC_CUP: "Cup",
        }.items():
            paths = [FROM_SCRACH_RETINOLOGISTS[r] / folder / (img + ".png") for img in images]
            if all(f.exists() for f in paths):
                dataset._data[lesion.value] = [path.absolute() for path in paths]
            else:
                print(f"Missing {lesion} for retinologist {r}.")
        for col in list(dataset._data.columns):
            if col.endswith("pre"):
                del dataset._data[col]

    return fromscratch_datasets


def load_reannotations():
    REFINED_ANNOTATION = ...
    REFINED_PREANNOTATION = Path("../PATH/TO/MAPLES-DR/preannotations-improved-apr24/annotations/")

    reannotated_dataset = [maples_dr.quick_api.GLOBAL_LOADER.load_dataset() for i in range(2)]

    for r in range(2):
        dataset = reannotated_dataset[r]

        # Set Refined annotation
        # TODO: once the refined annotations are available, set the paths here

        # Remove all preannotations
        for col in list(dataset._data.columns):
            if col.endswith("pre"):
                del dataset._data[col]

        # Load refined preannotations
        for lesion, folder in {
            Bio.MICROANEURYSMS: "Microaneurysms",
            Bio.HEMORRHAGES: "Hemorrhages",
            Bio.EXUDATES: "Exudates",
            Bio.COTTON_WOOL_SPOTS: "CottonWoolSpots",
        }.items():
            paths = [REFINED_PREANNOTATION / folder / (img + ".png") for img in dataset.keys()]
            if all(f.exists() for f in paths):
                dataset._data[lesion.value + "_pre"] = [path.absolute() for path in paths]
            else:
                print(f"Missing preannotation for {lesion}.")

    return reannotated_dataset


def count_lesions(dataset: maples_dr.dataset.Dataset, biomarker: maples_dr.dataset.BiomarkerField, preannotation=False):
    counts = []
    for sample in dataset:
        bio = sample.read_biomarker(biomarker, image_format="bgr", pre_annotation=preannotation)
        if bio.any():
            counts.append(np.max(label(bio)))
        else:
            counts.append(0)
    return pd.Series(counts, index=dataset.data.index)
