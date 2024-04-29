import logging

import maples_dr
from maples_dr.dataset import BiomarkerField as Bio
from maples_dr.dataset import DiagnosisField
from pytest import fixture

LOGGER = logging.getLogger(__name__)


@fixture
def local_dataset():
    local_loader = maples_dr.loader.DatasetLoader()
    local_loader.configure(
        maples_dr_path="examples/PATH/TO/MAPLES-DR/AdditionalData.zip",
    )
    return local_loader.load_dataset("all_with_duplicates")


def test_download(local_dataset):
    maples_dr.clear_download_cache()

    downloaded_dataset = maples_dr.load_dataset("all_with_duplicates")

    # === Check data content ===
    # Biomarkers
    for local_sample, downloaded_sample in zip(local_dataset, downloaded_dataset, strict=True):
        for biomarker in Bio:
            assert (
                local_sample[biomarker] == downloaded_sample[biomarker]
            ), f"Inconsistent {biomarker} for sample {local_sample.name}."

        for field in DiagnosisField:
            assert local_sample[field] == downloaded_sample[field], (
                f"Inconsistent {field} for sample {local_sample.name}."
                f" (local: {local_sample[field]}, downloaded: {downloaded_sample[field]})",
            )
