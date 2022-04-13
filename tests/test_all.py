def test_split():
    from re_nobm_pcc import preprocessing
    train = preprocessing.train.sizes['pxl']
    validate = preprocessing.validate.sizes['pxl']
    test = preprocessing.test.sizes['pxl']
    assert (train > validate) and (validate > test)


def test_perceptron_model():
    from re_nobm_pcc import perceptron
    assert perceptron.Full().built
    assert perceptron.Reduced().built
