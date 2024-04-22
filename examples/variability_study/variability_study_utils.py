from pathlib import Path

import maples_dr
import networkx as nx
import numpy as np
import pandas as pd
from jppype.utilities.geometric import Point
from maples_dr.dataset import BiomarkerField as Bio
from skimage.measure import label, regionprops


def centroid(map):
    props = regionprops(map.astype(np.uint8))
    return Point(*props[0].centroid)


def regions_f1(map1, map2):
    labels1 = label(map1)
    labels2 = label(map2)

    nb_l1 = np.max(labels1)
    nb_l2 = np.max(labels2)

    sen = 0.0
    for l1 in range(1, nb_l1 + 1):
        if np.any((labels1 == l1) & (labels2 > 0)):
            sen += 1
    if nb_l1 > 0:
        sen /= nb_l1
    else:
        sen = float("nan")

    spe = 0.0
    for l2 in range(1, nb_l2 + 1):
        if np.any((labels2 == l2) & (labels1 > 0)):
            spe += 1
    if nb_l2 > 0:
        spe /= nb_l2
    else:
        spe = float("nan")

    f1 = 2 * sen * spe / (sen + spe)
    return f1


def load_annotations_from_scratch(images=None):
    FROM_SCRATCH = Path("/home/gaby/These/Data/Fundus/MESSIDOR1500/FromScratch/")
    FROM_SCRACH_RETINOLOGISTS = [
        FROM_SCRATCH / "Marie-Carole Boucher",
        FROM_SCRATCH / "Micheal Brent",
        FROM_SCRATCH / "Renaud Duval",
    ]

    if images is None:
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


def load_new_annotations():
    REFINED_ANNOTATION = [
        Path("../PATH/TO/MAPLES-DR/Variability/Daniel"),
        Path("../PATH/TO/MAPLES-DR/Variability/Fares_resized"),
    ]
    REFINED_PREANNOTATION = Path("../PATH/TO/MAPLES-DR/preannotations-improved-apr24/annotations_resized/")

    images = [_.stem for _ in (REFINED_ANNOTATION[0] / "Microaneurysms").rglob("*.png")]

    def create_dataset():
        return maples_dr.quick_api.GLOBAL_LOADER.load_dataset()[images]

    new_datasets = [create_dataset() for p in REFINED_ANNOTATION]

    for r, (path, dataset) in enumerate(zip(REFINED_ANNOTATION, new_datasets, strict=True)):
        for lesion, folder in {
            Bio.MICROANEURYSMS: "Microaneurysms",
            Bio.HEMORRHAGES: "Hemorrhages",
            Bio.RED_UNCERTAINS: "RedUncertains",
            Bio.EXUDATES: "Exudates",
            Bio.COTTON_WOOL_SPOTS: "CottonWoolSpots",
            Bio.DRUSENS: "Drusens",
            Bio.BRIGHT_UNCERTAINS: "BrightUncertains",
        }.items():
            paths = [path / folder / (img + ".png") for img in images]
            if all(f.exists() for f in paths):
                dataset._data[lesion.value] = [path.absolute() for path in paths]
            else:
                print(f"Missing {lesion} for retinologist {r}.")
        for col in list(dataset._data.columns):
            if col.endswith("pre"):
                del dataset._data[col]

        # Remove all preannotations
        for col in list(dataset._data.columns):
            if col.endswith("pre"):
                del dataset._data[col]

        # Replace by refined preannotations
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

    return new_datasets


def load_refined_preannotations(resized=True):
    REFINED_PREANNOTATION = Path("../PATH/TO/MAPLES-DR/preannotations-improved-apr24/annotations_resized/")
    dataset = maples_dr.quick_api.GLOBAL_LOADER.load_dataset()

    # Replace by refined preannotations
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

    return dataset


def count_lesions(dataset: maples_dr.dataset.Dataset, biomarker: maples_dr.dataset.BiomarkerField, preannotation=False):
    counts = []
    for sample in dataset:
        bio = sample.read_biomarker(biomarker, image_format="bgr", pre_annotation=preannotation)
        if bio.any():
            counts.append(np.max(label(bio)))
        else:
            counts.append(0)
    return pd.Series(counts, index=dataset.data.index)


def multi_annotator_regions_diff(initial_map, *edited_maps):
    """Compute the difference between an initial map and a set of edited maps.

    For each region (connected component) in the initial map, the function determines whether it was removed, changed,
    or kept in each of the edited maps. It also determines whether new regions were added in the edited maps.
    The function returns a map of regions label and a dictionary indicating weither each region was removed, changed,
    kept, or added in each edited map.

    When new regions added in several edited maps overlap, they are merged and considered as a single region.

    Parameters
    ----------
    initial_map : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    initial_map_labels = label(initial_map)
    n_initial_labels = np.max(initial_map_labels)

    labels = []
    region_diffs = {_: [] for _ in range(1, n_initial_labels + 1)}
    added_graph = nx.Graph()

    for i, m in enumerate(edited_maps):
        labels_m, removed_m, changed_m, kept_m, added_m = regions_diff(initial_map_labels, m)
        for l in removed_m:
            region_diffs[l].append("R")
        for l in changed_m:
            region_diffs[l].append("C")
        for l in kept_m:
            region_diffs[l].append("K")

        # Create node for added regions
        for l in added_m:
            added_graph.add_node((i, l))
            for i_previous_labels, previous_labels in enumerate(labels):
                overlapping_l = set(np.unique(previous_labels[labels_m == l])) - {0}
                for ol in overlapping_l:
                    added_graph.add_edge((i, l), (i_previous_labels, ol))

        labels.append(labels_m)

    # Solve connected components in the added graph
    added_components = list(nx.connected_components(added_graph))
    final_labels = initial_map_labels.copy()
    l_added = n_initial_labels
    for component in added_components:
        l_added += 1
        for i, l in component:
            final_labels[labels[i] == l] = l_added

        region_diffs[l_added] = ["A" if i in {_[0] for _ in component} else "" for i in range(len(edited_maps))]

    return final_labels, region_diffs


def regions_diff(initial_label_map, map_after):
    """Compute the difference between two maps in terms of connected components.

    Parameters
    ----------
    map_before :
        A binary numpy array representing the initial state of the map.
    map_after :
        A binary numpy array representing the final state of the map.

    Returns
    -------
    removed :
        A numpy array of the same shape as the input maps, where each pixel is labeled with the index of the connected
        component that was removed.
    added :
        A numpy array of the same shape as the input maps, where each pixel is labeled with the index of the connected
        component that was added.
    changed :
        A numpy array of the same shape as the input maps, where each pixel is labeled with the index of the connected
        component that was changed.
    kept :
        A numpy array of the same shape as the input maps, where each pixel is labeled with the index of the connected
        component that was kept.
    """
    labels1 = initial_label_map
    if labels1.dtype == bool:
        labels1 = label(labels1)
    labels2 = label(map_after)

    removed_l1 = set()
    changed_l1 = set()
    kept_l1 = set(np.unique(labels1)) - {0}
    labels = labels1.copy()

    added_l2 = set(range(1, np.max(labels2) + 1))

    for l1 in list(kept_l1):
        l2s = np.unique(labels2[labels1 == l1])

        if list(l2s) == [0]:
            # The connected component was removed
            kept_l1.remove(l1)
            removed_l1.add(l1)
            continue
        else:
            if len(l2s) == 1:
                l2 = l2s[0]
                if np.all((labels2 == l2) == (labels1 == l1)):
                    # The connected component was kept unchanged
                    added_l2.remove(l2)
                    continue

        # The connected component was changed
        changed_l1.add(l1)
        kept_l1.remove(l1)
        added_l2 -= set(l2s)

    added_l = np.max(labels)
    added_l1 = set()
    for l2 in added_l2:
        added_l += 1
        labels[(labels2 == l2) & (labels == 0)] = added_l
        added_l1.add(added_l)

    return labels, removed_l1, changed_l1, kept_l1, added_l1
