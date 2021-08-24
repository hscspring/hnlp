from dataclasses import dataclass
import torch
from hnlp.converter.converter_pt import convert_model_input


def test_convert_model_input():
    @dataclass
    class A:
        @convert_model_input
        def __call__(self, inputs):
            return inputs

    a = A()
    x = [[1, 2], [3, 4]]
    inp = a(x)
    assert isinstance(inp["input_ids"], torch.Tensor) == True
    assert inp["input_ids"].shape == torch.Size([2, 2])


def test_convert_model_input_with_parameter():
    @dataclass
    class A:
        @convert_model_input(target="pretrained")
        def __call__(self, inputs):
            return inputs

    a = A()
    x = [[1, 2], [3, 4]]
    inp = a(x)
    assert isinstance(inp["input_ids"], torch.Tensor) == True
    assert inp["input_ids"].shape == torch.Size([2, 2])


def test_convert_model_input_with_not_implemented():
    @dataclass
    class A:
        @convert_model_input(target="other")
        def __call__(self, inputs):
            return inputs

    a = A()
    x = [[1, 2], [3, 4]]
    inp = a(x)
    assert type(inp) == list
    assert inp == x


def test_convert_model_input_with_one_label():
    @dataclass
    class A:
        @convert_model_input
        def fit(self, inputs, labels):
            return inputs["input_ids"] * labels

    a = A()
    x = [[1, 2], [3, 4]]
    y = [1, 0]
    yp = a.fit(x, y)
    assert isinstance(yp, torch.Tensor) == True
    assert yp.shape == torch.Size([2, 2])
    assert yp.sum_to_size(1).tolist()[0] == 4


def test_convert_model_input_with_one_label_str():
    @dataclass
    class A:
        @convert_model_input
        def fit(self, inputs, labels):
            return inputs["input_ids"] * labels

    a = A()
    x = [[1, 2], [3, 4]]
    y = ["0", "1"]
    yp = a.fit(x, y)
    assert isinstance(yp, torch.Tensor) == True
    assert yp.shape == torch.Size([2, 2])
    assert yp.sum_to_size(1).tolist()[0] == 6


def test_convert_model_input_with_none():
    @dataclass
    class A:
        @convert_model_input
        def fit(self, inputs, labels):
            return inputs["input_ids"] * labels

    a = A()
    try:
        a.fit()
    except Exception as e:
        assert "Invalid inputs" in str(e)
