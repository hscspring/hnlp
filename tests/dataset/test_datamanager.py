import pytest
from hnlp.dataset.datamanager import MapStyleDataset


@pytest.fixture
def input_data():
    return [["I", "love", "you", ",", "and", "you", "love", "me", "."], [1, 2, 3, 4, 5]]


def test_mapstyle_dataset(input_data):
    ms = MapStyleDataset(input_data, 1, 512)
    assert ms.data == input_data
    assert ms[0] == ["I", "love", "you", ",", "and", "you", "love", "me", "."]
    assert len(ms) == 2


def test_mapstyle_dataset_min_len(input_data):
    ms = MapStyleDataset(input_data, 6, 512)
    assert ms.data == [["I", "love", "you", ",", "and", "you", "love", "me", "."]]
    assert len(ms.data) == 1
