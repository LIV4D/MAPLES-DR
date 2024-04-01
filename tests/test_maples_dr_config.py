from maples_dr.config import DatasetConfig


def test_config():
    cfg1 = DatasetConfig(resize=True)
    assert cfg1.resize is True

    cfg2 = DatasetConfig(resize=512)
    assert cfg2.resize == 512

    cfg1.update(dict(resize=200))
    assert cfg1.resize == 200
