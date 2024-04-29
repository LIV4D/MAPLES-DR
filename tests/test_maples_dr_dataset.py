import logging

import maples_dr
import numpy as np
from maples_dr.dataset import (
    AGGREGATED_BIOMARKERS,
    BiomarkersAnnotationInfos,
    BiomarkersAnnotationTasks,
    DiagnosisField,
)
from maples_dr.dataset import BiomarkerField as Bio
from pytest import fixture

LOGGER = logging.getLogger(__name__)


@fixture
def train_set():
    maples_dr.configure(
        maples_dr_path="examples/PATH/TO/MAPLES-DR/AdditionalData.zip",
    )
    return maples_dr.load_train_set()


@fixture
def test_set():
    maples_dr.configure(
        maples_dr_path="examples/PATH/TO/MAPLES-DR/AdditionalData.zip",
    )
    return maples_dr.load_test_set()


@fixture
def all_set():
    maples_dr.configure(
        maples_dr_path="examples/PATH/TO/MAPLES-DR/AdditionalData.zip",
    )
    return maples_dr.quick_api.GLOBAL_LOADER.load_dataset("all_with_duplicates")


def test_dataset_properties(train_set, test_set, all_set):
    assert len(train_set) == 138
    assert len(test_set) == 60
    assert len(all_set) == 200

    # === Check data content ===
    # Biomarkers
    for biomarker in Bio:
        if biomarker not in AGGREGATED_BIOMARKERS:
            assert biomarker.value in all_set.data
    for biomarker in [
        Bio.EXUDATES,
        Bio.HEMORRHAGES,
        Bio.MICROANEURYSMS,
        Bio.VESSELS,
    ]:
        assert f"{biomarker.value}_pre" in all_set.data

    # Biomarkers tasks
    for task in BiomarkersAnnotationTasks:
        for infos in BiomarkersAnnotationInfos:
            assert f"{task.value}_{infos.value}" in all_set.data

    # Diagnosis
    for r in "ABC":
        for pathology in ("dr", "me"):
            grades = ["R0", "R1", "R2", "R3", "R4A", "R4S", "R6"] if pathology == "dr" else ["M0", "M1", "M2", "M6"]

            assert pathology in all_set.data
            invalid_values = set(all_set.data[pathology].unique()) - set(grades)
            assert not invalid_values, f"Invalid values for {pathology}: {invalid_values}"

            assert f"{pathology}_{r}" in all_set.data
            invalid_values = set(all_set.data[f"{pathology}_{r}"].unique()) - set(grades)
            assert not invalid_values, f"Invalid values for {pathology}_{r}: {invalid_values}"

        assert f"dr_{r}_comment" in all_set.data


def test_dataset_numerical_indexing(all_set):
    names0 = all_set.keys()

    assert all_set[0].name == names0[0]
    assert all_set[-1].name == names0[-1]
    assert all_set[:5].keys() == names0[:5]
    assert all_set[5:10].keys() == names0[5:10]
    assert all_set[-5:].keys() == names0[-5:]


def test_dataset_samples(all_set):
    sample = all_set[0]
    sample._cfg.image_format = "bgr"

    assert sample[Bio.VESSELS].sum() > 200

    # Test equivalence of __getitem__ and actual read methods for biomarkers
    for biomarker in Bio:
        assert np.all(sample[biomarker] == sample.read_biomarker(biomarker))
        assert sample.read_biomarker(biomarker, pre_annotation=True) is not None

        # Test aggregated biomarkers
        if biomarker in AGGREGATED_BIOMARKERS:
            no_diff = sample[biomarker] == np.any(
                [sample[bio] for bio in AGGREGATED_BIOMARKERS[biomarker]],
                axis=0,
            )
            assert np.all(no_diff)

    # Test field name tolerance
    assert np.all(sample["CottonWoolSpots"] == sample["cotton_wool_spots"])
    assert np.all(sample["red"] == sample["RedLesions"])
    assert np.all(sample["bright"] == sample["bright_lesions"])
    assert np.all(sample["CUP"] == sample["OPTIC_CUP"])
    assert np.all(sample["Disk"] == sample["opticDisk"])

    # Test existence of diagnosis fields
    assert sample[DiagnosisField.DR] in ("R0", "R1", "R2", "R3", "R4A")
    assert sample[DiagnosisField.ME] in ("M0", "M1", "M2")

    # Test existence of diagnosis infos fields
    for r in "ABC":
        for pathology in ("dr", "me"):
            assert pathology in all_set.data
            assert f"{pathology}_{r}" in all_set.data
        assert f"dr_{r}_comment" in all_set.data


def test_anatomical_structures_on_all_samples(all_set, caplog):
    maples_dr.configure(image_format="bgr")
    missing_vessels = set()
    missing_disc = set()
    missing_cup = set()
    missing_macula = set()
    for sample in all_set:
        if sample[Bio.VESSELS].sum() == 0:
            missing_vessels.add(sample.name)
        if sample[Bio.OPTIC_DISC].sum() == 0:
            missing_disc.add(sample.name)

        caplog.clear()
        if sample[Bio.MACULA].sum() == 0:
            if sample.name not in maples_dr.quick_api.GLOBAL_LOADER.dataset_record.get("no_macula", []):
                missing_macula.add(sample.name)
            else:
                assert (
                    f"The macula is not segmented on image {sample.name}!" in caplog.text
                ), "Missing macula should be logged."
        if sample[Bio.OPTIC_CUP].sum() == 0:
            if sample.name not in maples_dr.quick_api.GLOBAL_LOADER.dataset_record.get("no_cup", []):
                missing_cup.add(sample.name)
            else:
                assert (
                    f"The optic cup is not segmented on image {sample.name}!" in caplog.text
                ), "Missing cup should be logged."

    assert not missing_vessels, f"Missing vessels for images {missing_vessels}."
    assert not missing_disc, f"Missing disc for images {missing_disc}."
    assert not missing_macula, f"Missing macula for images {missing_macula}."
    assert not missing_cup, f"Missing cup for images {missing_cup}."


def test_lesions_on_all_pathological_samples(all_set):
    maples_dr.configure(image_format="bgr")

    missing_lesions = set()
    for sample in all_set:
        if sample["dr"] not in ("R0", "R6") or sample["me"] not in ("M0", "M6"):
            if sample.read_biomarker([Bio.RED_LESIONS, Bio.BRIGHT_LESIONS]).sum() == 0:
                missing_lesions.add(sample.name)

    assert not missing_lesions, f"Missing lesions for {len(missing_lesions)} images {missing_lesions}."


def test_exclusion_of_missing_biomarkers(all_set):
    dataset_record = maples_dr.quick_api.GLOBAL_LOADER.dataset_record
    no_macula = set(dataset_record.get("no_macula", []))
    no_cup = set(dataset_record.get("no_cup", []))
    all_names = set(all_set.keys())

    maples_dr.configure(exclude_missing_cup=True)
    dataset_no_cup = maples_dr.quick_api.GLOBAL_LOADER.load_dataset("all_with_duplicates")
    no_cup_names = set(_.name for _ in dataset_no_cup)
    assert no_cup_names == all_names - no_cup, "Images with missing cup should be excluded."

    maples_dr.configure(exclude_missing_macula=True, exclude_missing_cup=False)
    dataset_no_macula = maples_dr.quick_api.GLOBAL_LOADER.load_dataset("all_with_duplicates")
    no_macula_names = set(_.name for _ in dataset_no_macula)
    assert no_macula_names == all_names - no_macula, "Images with missing macula should be excluded."

    maples_dr.configure(exclude_missing_macula=True, exclude_missing_cup=True)
    dataset_no_lesions = maples_dr.quick_api.GLOBAL_LOADER.load_dataset("all_with_duplicates")
    no_lesions_names = set(_.name for _ in dataset_no_lesions)
    assert no_lesions_names == all_names - no_macula - no_cup, "Images with missing macula and cup should be excluded."

    maples_dr.configure(exclude_missing_macula=False, exclude_missing_cup=False)
    dataset_all = maples_dr.quick_api.GLOBAL_LOADER.load_dataset("all_with_duplicates")
    all_names2 = set(_.name for _ in dataset_all)
    assert all_names2 == all_names, "All images should be included back when disabling exclusion."
