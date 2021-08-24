from hnlp.dataset.preprocessor import Preprocessor


def test_preprocessor_single_text():
    preprocessor = Preprocessor()
    text = "爱情"
    assert preprocessor(text) == text


def test_preprocessor_single_text_tuple_without_label():
    preprocessor = Preprocessor()
    text = ("爱情",)
    assert preprocessor(text) == text


def test_preprocessor_single_text_with_label():
    preprocessor = Preprocessor()
    text = ("爱情", 1)
    assert preprocessor(text) == text


def test_preprocessor_multiple_texts():
    preprocessor = Preprocessor()
    text_list = ["我喜欢你", "你也喜欢我。"]
    assert preprocessor(text_list) == text_list


def test_preprocessor_multiple_texts_tuple_without_labels():
    preprocessor = Preprocessor()
    text_list = [("我喜欢你",), ("你也喜欢我。",)]
    assert preprocessor(text_list) == text_list


def test_preprocessor_multiple_texts_with_labels():
    preprocessor = Preprocessor()
    text_list = [("我喜欢你", "1"), ("你也喜欢我。", "1")]
    assert preprocessor(text_list) == text_list


def test_preprocessor_single_element():
    preprocessor = Preprocessor(pats=["emj"])
    text = ["😁哈哈"]
    assert preprocessor(text) == ["哈哈"]


def test_preprocessor_single_element_tuple_without_label():
    preprocessor = Preprocessor(pats=["emj"])
    text = [("😁哈哈",)]
    assert preprocessor(text) == [("哈哈",)]


def test_preprocessor_single_element_with_label():
    preprocessor = Preprocessor(pats=["emj"])
    text = [("😁哈哈", "0")]
    assert preprocessor(text) == [("哈哈", "0")]
