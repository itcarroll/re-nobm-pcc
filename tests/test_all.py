def test_split():
    from scripts import preprocessing
    train = preprocessing.train.sizes['pxl']
    validate = preprocessing.validate.sizes['pxl']
    test = preprocessing.test.sizes['pxl']
    assert (train > validate) and (validate > test)


def test_perceptron_model():
    from scripts import perceptron
    assert perceptron.Full().built
    assert perceptron.Reduced().built
