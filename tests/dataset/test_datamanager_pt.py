import pytest
from hnlp.dataset.datamanager_pt import DataManagerPt


@pytest.fixture
def input_data():
    return [["I", "love", "you", ",", "and", "you", "love", "me", "."], [1, 2, 3, 4, 5]]


def test_data_manager_sequence_default(input_data):
    # batch_size default is 1
    manager = DataManagerPt(name="sequence", dynamic_length=True)
    data = manager(input_data)
    result = list(data)
    assert result == [
        [["I", "love", "you", ",", "and", "you", "love", "me", "."]],
        [[1, 2, 3, 4, 5]],
    ]


def test_data_manager_random_default(input_data):
    # batch_size default is 1
    manager = DataManagerPt(name="random", dynamic_length=True)
    data = manager(input_data)
    result = list(data)
    assert result == [
        [["I", "love", "you", ",", "and", "you", "love", "me", "."]],
        [[1, 2, 3, 4, 5]],
    ] or result == [
        [[1, 2, 3, 4, 5]],
        [["I", "love", "you", ",", "and", "you", "love", "me", "."]],
    ]


def test_data_manager_sequence_pad(input_data):
    # min_seq_len default is 1
    manager = DataManagerPt(batch_size=3, name="sequence", dynamic_length=True)
    data = manager(input_data)
    result = list(data)
    assert result == [
        [
            ["I", "love", "you", ",", "and", "you", "love", "me", "."],
            [1, 2, 3, 4, 5, 0, 0, 0, 0],
        ]
    ]


def test_data_manager_random_pad(input_data):
    # min_seq_len default is 1
    manager = DataManagerPt(batch_size=3, name="random", dynamic_length=True)
    data = manager(input_data)
    result = list(data)
    assert result == [
        [
            [1, 2, 3, 4, 5, 0, 0, 0, 0],
            ["I", "love", "you", ",", "and", "you", "love", "me", "."],
        ]
    ] or result == [
        [
            ["I", "love", "you", ",", "and", "you", "love", "me", "."],
            [1, 2, 3, 4, 5, 0, 0, 0, 0],
        ]
    ]


def test_data_manager_sequence_seq_len(input_data):
    manager = DataManagerPt(min_seq_len=6, batch_size=3, name="sequence", dynamic_length=True)
    data = manager(input_data)
    assert list(data) == [[["I", "love", "you", ",", "and", "you", "love", "me", "."]]]


def test_data_manager_random_seq_len(input_data):
    manager = DataManagerPt(min_seq_len=6, batch_size=3, name="random", dynamic_length=True)
    data = manager(input_data)
    assert list(data) == [[["I", "love", "you", ",", "and", "you", "love", "me", "."]]]


def test_data_manager_dynamic_length(input_data):
    manager = DataManagerPt(name="sequence", dynamic_length=False, max_seq_len=9)
    data = manager(input_data)
    assert list(data) == [
        [["I", "love", "you", ",", "and", "you", "love", "me", "."]],
        [[1, 2, 3, 4, 5, 0, 0, 0, 0]],
    ]


def test_data_manager_sequence_drop_last(input_data):
    input_data = input_data + [["1", "2", "3", "4", "5"]]
    manager = DataManagerPt(drop_last=True, batch_size=2, name="sequence", dynamic_length=True)
    data = manager(input_data)
    result = list(data)
    assert result == [
        [
            [1, 2, 3, 4, 5, 0, 0, 0, 0],
            ["I", "love", "you", ",", "and", "you", "love", "me", "."],
        ]
    ] or result == [
        [
            ["I", "love", "you", ",", "and", "you", "love", "me", "."],
            [1, 2, 3, 4, 5, 0, 0, 0, 0],
        ]
    ]


def test_data_manager_random_drop_last(input_data):
    input_data = input_data + [["1", "2", "3", "4", "5"]]
    manager = DataManagerPt(drop_last=True, batch_size=2, name="random")
    data = manager(input_data)
    result = list(data)
    assert len(result[0][0]) == len(result[0][1])
