import logging
from time import time

import maples_dr
import numpy as np
from maples_dr.dataset import (
    BiomarkerField,
)
from pytest import fixture

LOGGER = logging.getLogger(__name__)


@fixture
def train_set():
    maples_dr.configure(
        maples_dr_path="examples/PATH/TO/MAPLES-DR/AdditionalData.zip",
        messidor_path="examples/PATH/TO/MESSIDOR/",
    )
    return maples_dr.load_train_set()


def test_cache_efficiency_fundus(train_set, tmp_path):
    N_SAMPLES = 50

    # Measure Timing without cache
    maples_dr.configure(cache=False, image_format="bgr")

    tic = time()
    fundus_nocache = [sample.read_fundus() for sample in train_set[:N_SAMPLES]]
    dt_nocache = time() - tic

    # Measure Timing with cache
    maples_dr.configure(cache=tmp_path)

    # A first time to create the cache
    tic = time()
    fundus_cache0 = [sample.read_fundus() for sample in train_set[:N_SAMPLES]]
    dt_cache0 = time() - tic

    # A second time to use the cache
    tic = time()
    fundus_cache1 = [sample.read_fundus() for sample in train_set[:N_SAMPLES]]
    dt_cache1 = time() - tic

    # A third time to check consistency
    tic = time()
    fundus_cache2 = [sample.read_fundus() for sample in train_set[:N_SAMPLES]]
    dt_cache2 = time() - tic

    # Check that the cache data is consistent
    def tolerable_diff(a, b):
        return np.sum(np.abs(a.astype(float) - b) > 3) < a.size / 100

    assert all(
        tolerable_diff(a, b) for a, b in zip(fundus_cache1, fundus_cache0, strict=True)
    ), "Cache data inconsistent between cache creation and usage."
    assert all(
        tolerable_diff(a, b) for a, b in zip(fundus_cache1, fundus_cache2, strict=True)
    ), "Cache data inconsistent between cache usage."
    assert all(
        tolerable_diff(a, b) for a, b in zip(fundus_cache1, fundus_nocache, strict=True)
    ), "Cache data inconsistent with standard read."

    # Check that the cache is time effective
    assert max(dt_cache1, dt_cache2) < dt_cache0
    assert max(dt_cache1, dt_cache2) < dt_nocache

    # Print timing information
    LOGGER.info(f"Without cache: {dt_nocache/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"With cache: {dt_cache1/N_SAMPLES*1000:.1f}ms , {dt_cache2/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"Cache creation: {dt_cache0/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"Cache efficiency: {dt_nocache / max(dt_cache1, dt_cache2):.2f}")


def test_cache_efficiency_vessels(train_set, tmp_path):
    N_SAMPLES = 50

    # === COMPARE TIMING WHEN NO PROCESSING IS APPLIED ===
    # Measure Timing without cache
    maples_dr.configure(cache=False, image_format="bgr")

    tic = time()
    vessels_nocache = [sample.read_biomarker(BiomarkerField.VESSELS) for sample in train_set[:N_SAMPLES]]
    dt_nocache = time() - tic

    # Measure Timing with cache
    maples_dr.configure(cache=tmp_path)
    maples_dr.clear_cache()
    tic = time()
    vessels_cache0 = [sample.read_biomarker(BiomarkerField.VESSELS) for sample in train_set[:N_SAMPLES]]
    dt_cache0 = time() - tic

    # Check that the cache data is consistent
    assert all(
        np.all(a == b) for a, b in zip(vessels_cache0, vessels_nocache, strict=True)
    ), "Cache data inconsistent with standard read."

    # Check that the cache is time effective
    assert (
        (dt_cache0 - dt_nocache) / dt_nocache < 1e-2
    ), f"Cache read is slower than no cache read (no cache: {dt_nocache:.2f}s, cache: {dt_cache0:.2f}s)"

    # Print timing information
    LOGGER.info("=== NO PROCESSING ===")
    LOGGER.info(f"Without cache: {dt_nocache/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"With cache: {dt_cache0/N_SAMPLES*1000:.1f}ms")

    # === COMPARE TIMING WHEN PROCESSING IS APPLIED ===
    maples_dr.configure(resize=512)

    # Measure Timing without cache
    maples_dr.configure(cache=False, image_format="bgr")
    tic = time()
    vessels_nocache = [sample.read_biomarker(BiomarkerField.VESSELS) for sample in train_set[:N_SAMPLES]]
    dt_nocache = time() - tic

    # Measure Timing with cache
    maples_dr.configure(cache=tmp_path)

    # A first time to create the cache
    tic = time()
    vessels_cache0 = [sample.read_biomarker(BiomarkerField.VESSELS) for sample in train_set[:N_SAMPLES]]
    dt_cache0 = time() - tic

    # A second time to use the cache
    tic = time()
    vessels_cache1 = [sample.read_biomarker(BiomarkerField.VESSELS) for sample in train_set[:N_SAMPLES]]
    dt_cache1 = time() - tic

    # A third time to check consistency
    tic = time()
    vessels_cache2 = [sample.read_biomarker(BiomarkerField.VESSELS) for sample in train_set[:N_SAMPLES]]
    dt_cache2 = time() - tic

    # Check that the cache data is consistent
    assert all(
        np.all(a == b) for a, b in zip(vessels_cache1, vessels_cache0, strict=True)
    ), "Cache data inconsistent between cache creation and usage."
    assert all(
        np.all(a == b) for a, b in zip(vessels_cache1, vessels_cache2, strict=True)
    ), "Cache data inconsistent between cache usage."
    assert all(
        np.all(a == b) for a, b in zip(vessels_cache1, vessels_nocache, strict=True)
    ), "Cache data inconsistent with standard read."

    # Check that the cache is time effective
    assert (
        max(dt_cache1, dt_cache2) < dt_cache0
    ), f"Cache read is slower than cache creation (create: {dt_cache0:.2f}s, read: {dt_cache1:.2f}s, {dt_cache2:.2f}s)"
    assert (
        max(dt_cache1, dt_cache2) < dt_nocache
    ), f"Cache read is slow (cache: {dt_cache1:.2f}s, {dt_cache2:.2f}s, no cache: {dt_nocache:.2f}s)"

    # Print timing information
    LOGGER.info("=== WITH PROCESSING ===")
    LOGGER.info(f"Without cache: {dt_nocache/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"With cache: {dt_cache1/N_SAMPLES*1000:.1f}ms , {dt_cache2/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"Cache creation: {dt_cache0/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"Cache efficiency: {dt_nocache / max(dt_cache1, dt_cache2):.2f}")


def test_cache_efficiency_red(train_set, tmp_path):
    N_SAMPLES = 50

    # === COMPARE TIMING WHEN NO PROCESSING IS APPLIED ===
    # Measure Timing without cache
    maples_dr.configure(cache=False, image_format="bgr")

    tic = time()
    red_nocache = [sample.read_biomarker(BiomarkerField.RED_LESIONS) for sample in train_set[:N_SAMPLES]]
    dt_nocache = time() - tic

    # Measure Timing with cache
    maples_dr.configure(cache=tmp_path)
    maples_dr.clear_cache()

    # A first time to create the cache
    tic = time()
    red_cache0 = [sample.read_biomarker(BiomarkerField.RED_LESIONS) for sample in train_set[:N_SAMPLES]]
    dt_cache0 = time() - tic

    # A second time to use the cache
    tic = time()
    red_cache1 = [sample.read_biomarker(BiomarkerField.RED_LESIONS) for sample in train_set[:N_SAMPLES]]
    dt_cache1 = time() - tic

    # Check that the cache data is consistent
    assert all(
        np.all(a == b) for a, b in zip(red_cache1, red_cache0, strict=True)
    ), "Cache data inconsistent between cache creation and usage."
    assert all(
        np.all(a == b) for a, b in zip(red_cache0, red_nocache, strict=True)
    ), "Cache data inconsistent with standard read."

    # Check that the cache is time effective
    assert (
        dt_cache1 < dt_cache0
    ), f"Cache read is slower than cache creation (create: {dt_cache0:.2f}s, read: {dt_cache1:.2f}s)"
    assert (
        (dt_cache1 - dt_nocache) / dt_nocache < 1e-2
    ), f"Cache read is slower than no cache read (no cache: {dt_nocache:.2f}s, cache: {dt_cache0:.2f}s)"

    # Print timing information
    LOGGER.info("=== NO PROCESSING ===")
    LOGGER.info(f"Without cache: {dt_nocache/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"With cache: {dt_cache1/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"Cache creation: {dt_cache0/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"Cache efficiency: {dt_nocache / dt_cache1:.2f}")

    # === COMPARE TIMING WHEN PROCESSING IS APPLIED ===
    maples_dr.configure(resize=512)

    # Measure Timing without cache
    maples_dr.configure(cache=False, image_format="bgr")
    tic = time()
    red_nocache = [sample.read_biomarker(BiomarkerField.RED_LESIONS) for sample in train_set[:N_SAMPLES]]
    dt_nocache = time() - tic

    # Measure Timing with cache
    maples_dr.configure(cache=tmp_path)

    # A first time to create the cache
    tic = time()
    red_cache0 = [sample.read_biomarker(BiomarkerField.RED_LESIONS) for sample in train_set[:N_SAMPLES]]
    dt_cache0 = time() - tic

    # A second time to use the cache
    tic = time()
    red_cache1 = [sample.read_biomarker(BiomarkerField.RED_LESIONS) for sample in train_set[:N_SAMPLES]]
    dt_cache1 = time() - tic

    # A third time to check consistency
    tic = time()
    red_cache2 = [sample.read_biomarker(BiomarkerField.RED_LESIONS) for sample in train_set[:N_SAMPLES]]
    dt_cache2 = time() - tic

    # Check that the cache data is consistent
    assert all(
        np.all(a == b) for a, b in zip(red_cache1, red_cache0, strict=True)
    ), "Cache data inconsistent between cache creation and usage."
    assert all(
        np.all(a == b) for a, b in zip(red_cache1, red_cache2, strict=True)
    ), "Cache data inconsistent between cache usage."
    assert all(
        np.all(a == b) for a, b in zip(red_cache1, red_nocache, strict=True)
    ), "Cache data inconsistent with standard read."

    # Check that the cache is time effective
    assert (
        max(dt_cache1, dt_cache2) < dt_cache0
    ), f"Cache read is slower than cache creation (create: {dt_cache0:.2f}s, read: {dt_cache1:.2f}s, {dt_cache2:.2f}s)"
    assert (
        max(dt_cache1, dt_cache2) < dt_nocache
    ), f"Cache read is slow (cache: {dt_cache1:.2f}s, {dt_cache2:.2f}s, no cache: {dt_nocache:.2f}s)"

    # Print timing information
    LOGGER.info("=== WITH PROCESSING ===")
    LOGGER.info(f"Without cache: {dt_nocache/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"With cache: {dt_cache1/N_SAMPLES*1000:.1f}ms , {dt_cache2/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"Cache creation: {dt_cache0/N_SAMPLES*1000:.1f}ms")
    LOGGER.info(f"Cache efficiency: {dt_nocache / max(dt_cache1, dt_cache2):.2f}")
