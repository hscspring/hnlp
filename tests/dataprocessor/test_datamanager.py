import pytest
from hnlp.dataprocessor.datamanager import DataManager, MapStyleDataset


@pytest.fixture
def input_data():
    return [
        ["I", "love", "you", ",", "and", "you", "love", "me", "."],
        [1, 2, 3, 4, 5]
    ]


def test_mapstyle_dataset(input_data):
    ms = MapStyleDataset(input_data, 1, 512)
    assert ms.data == input_data
    assert ms[0] == ["I", "love", "you", ",", "and", "you", "love", "me", "."]
    assert len(ms) == 2


def test_mapstyle_dataset_min_len(input_data):
    ms = MapStyleDataset(input_data, 6, 512)
    assert ms.data == [["I", "love", "you", ",", "and", "you", "love", "me", "."]]
    assert len(ms) == 1


def test_data_manager_sequence_default(input_data):
    # batch_size default is 1
    manager = DataManager(name="sequence")
    data = manager(input_data)
    result = list(data)
    assert result == [
        [["I", "love", "you", ",", "and", "you", "love", "me", "."]],
        [[1, 2, 3, 4, 5]]
    ]


def test_data_manager_random_default(input_data):
    # batch_size default is 1
    manager = DataManager(name="random")
    data = manager(input_data)
    result = list(data)
    assert result == [
        [["I", "love", "you", ",", "and", "you", "love", "me", "."]],
        [[1, 2, 3, 4, 5]]
    ] or result == [
        [[1, 2, 3, 4, 5]],
        [["I", "love", "you", ",", "and", "you", "love", "me", "."]]
    ]


def test_data_manager_sequence_pad(input_data):
    # min_seq_len default is 1
    manager = DataManager(batch_size=3, name="sequence")
    data = manager(input_data)
    result = list(data)
    assert result == [[
        ["I", "love", "you", ",", "and", "you", "love", "me", "."],
        [1, 2, 3, 4, 5, 0, 0, 0, 0]
    ]]


def test_data_manager_random_pad(input_data):
    # min_seq_len default is 1
    manager = DataManager(batch_size=3, name="random")
    data = manager(input_data)
    result = list(data)
    assert result == [[
        [1, 2, 3, 4, 5, 0, 0, 0, 0],
        ["I", "love", "you", ",", "and", "you", "love", "me", "."]
    ]] or result == [[
        ["I", "love", "you", ",", "and", "you", "love", "me", "."],
        [1, 2, 3, 4, 5, 0, 0, 0, 0]
    ]]


def test_data_manager_sequence_seq_len(input_data):
    manager = DataManager(min_seq_len=6, batch_size=3, name="sequence")
    data = manager(input_data)
    assert list(data) == [[
        ["I", "love", "you", ",", "and", "you", "love", "me", "."]
    ]]


def test_data_manager_random_seq_len(input_data):
    manager = DataManager(min_seq_len=6, batch_size=3, name="random")
    data = manager(input_data)
    assert list(data) == [[
        ["I", "love", "you", ",", "and", "you", "love", "me", "."]
    ]]


def test_data_manager_sequence_drop_last(input_data):
    input_data = input_data + [["1", "2", "3", "4", "5"]]
    manager = DataManager(drop_last=True, batch_size=2, name="sequence")
    data = manager(input_data)
    result = list(data)
    assert result == [[
        [1, 2, 3, 4, 5, 0, 0, 0, 0],
        ["I", "love", "you", ",", "and", "you", "love", "me", "."]
    ]] or result == [[
        ["I", "love", "you", ",", "and", "you", "love", "me", "."],
        [1, 2, 3, 4, 5, 0, 0, 0, 0]
    ]]


def test_data_manager_random_drop_last(input_data):
    input_data = input_data + [["1", "2", "3", "4", "5"]]
    manager = DataManager(drop_last=True, batch_size=2, name="random")
    data = manager(input_data)
    result = list(data)
    assert len(result[0][0]) == len(result[0][1])
