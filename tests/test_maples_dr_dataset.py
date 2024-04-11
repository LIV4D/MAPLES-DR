import maples_dr
import numpy as np
from maples_dr.dataset import (
    AGGREGATED_BIOMARKERS,
    BiomarkerField,
    BiomarkersAnnotationInfos,
    BiomarkersAnnotationTasks,
    DiagnosisField,
    FundusField,
)
from pytest import fixture


@fixture
def train_set():
    maples_dr.configure(
        maples_dr_path="examples/PATH/TO/MAPLES-DR/AdditionalData.zip",
        messidor_path="examples/PATH/TO/MESSIDOR/",
    )
    return maples_dr.load_train_set()


def test_dataset_properties(train_set):
    assert len(train_set) == 138

    # === Check data content ===
    assert "fundus" in train_set.data

    # Biomarkers
    for biomarker in BiomarkerField:
        if biomarker not in AGGREGATED_BIOMARKERS:
            assert biomarker.value in train_set.data
    for biomarker in [
        BiomarkerField.EXUDATES,
        BiomarkerField.HEMORRHAGES,
        BiomarkerField.MICROANEURYSMS,
        BiomarkerField.VESSELS,
    ]:
        assert f"{biomarker.value}_pre" in train_set.data

    # Biomarkers tasks
    for task in BiomarkersAnnotationTasks:
        for infos in BiomarkersAnnotationInfos:
            assert f"{task.value}_{infos.value}" in train_set.data

    # Diagnosis
    for r in "ABC":
        for pathology in ("dr", "me"):
            grades = ["R0", "R1", "R2", "R3", "R4A", "R4S", "R6"] if pathology == "dr" else ["M0", "M1", "M2", "M6"]

            assert pathology in train_set.data
            invalid_values = set(train_set.data[pathology].unique()) - set(grades)
            assert not invalid_values, f"Invalid values for {pathology}: {invalid_values}"

            assert f"{pathology}_{r}" in train_set.data
            invalid_values = set(train_set.data[f"{pathology}_{r}"].unique()) - set(grades)
            assert not invalid_values, f"Invalid values for {pathology}_{r}: {invalid_values}"

        assert f"dr_{r}_comment" in train_set.data


def test_dataset_samples(train_set):
    sample = train_set[0]
    sample._cfg.image_format = "bgr"

    assert sample[BiomarkerField.VESSELS].sum() > 200

    # Test equivalence of __getitem__ and actual read methods for fundus and mask
    assert np.all(sample.read_fundus(preprocess=False) == sample[FundusField.RAW_FUNDUS])
    assert np.all(sample.read_roi_mask() == sample[FundusField.MASK])

    # Test equivalence of __getitem__ and actual read methods for biomarkers
    for biomarker in BiomarkerField:
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
            assert pathology in train_set.data
            assert f"{pathology}_{r}" in train_set.data
        assert f"dr_{r}_comment" in train_set.data


def test_cached_samples(train_set):
    maples_dr.configure(cache="examples/PATH/TO/CACHE")
    test_dataset_samples(train_set)
