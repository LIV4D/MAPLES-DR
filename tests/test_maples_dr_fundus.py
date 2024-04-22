import maples_dr
import numpy as np
from maples_dr.dataset import FundusField as FF
from pytest import fixture


@fixture
def all_set():
    maples_dr.configure(
        maples_dr_path="examples/PATH/TO/MAPLES-DR/AdditionalData.zip",
        messidor_path="examples/PATH/TO/MESSIDOR/",
    )
    return maples_dr.quick_api.GLOBAL_LOADER.load_dataset("all_with_duplicates")


def test_fundus(all_set):
    # === Check data content ===
    assert "fundus" in all_set.data

    sample = all_set[0]
    sample._cfg.image_format = "bgr"

    # Test equivalence of __getitem__ and actual read methods for fundus and mask
    assert np.all(sample.read_fundus(preprocess=False) == sample[FF.RAW_FUNDUS])
    assert np.all(sample.read_roi_mask() == sample[FF.MASK])
