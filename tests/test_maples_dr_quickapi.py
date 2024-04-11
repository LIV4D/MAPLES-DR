import maples_dr


def test_configure():
    maples_dr.configure(
        maples_dr_path="examples/PATH/TO/MAPLES-DR/AdditionalData.zip",
        messidor_path="examples/PATH/TO/MESSIDOR/",
        resize=512,
    )

    test_set = maples_dr.load_test_set()
    assert test_set._cfg.resize == 512

    train_set = maples_dr.load_train_set()
    assert test_set._cfg is train_set._cfg

    maples_dr.configure(resize=1024)
    assert test_set._cfg.resize == 1024
