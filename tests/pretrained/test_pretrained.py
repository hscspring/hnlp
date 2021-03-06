import torch
from torch import Tensor
from hnlp.pretrained.model import Pretrained
from hnlp.pretrained.processor import PretrainedProcessor
from hnlp.node import N
import pnlp


config = pnlp.read_yaml("tests/config.yaml")


def test_bert_pretrained_not_training():
    # (batch_size, seq_len)
    batch_size, seq_len = 8, 10
    model_path = config.get("pretrained").get("bert")

    input_ids = torch.randint(100, 1000, (batch_size, seq_len)).tolist()

    pretrained_processor = PretrainedProcessor()
    pretrained_model = Pretrained("bert", model_path, False)

    outputs = pretrained_model(pretrained_processor(input_ids))

    for output in outputs:
        assert len(output) == 2
        assert output[0].shape == torch.Size(
            [batch_size, seq_len, pretrained_model.node.model.config.hidden_size])
        assert output[1].shape == torch.Size(
            [batch_size, pretrained_model.node.model.config.hidden_size])


def test_bert_pretrained():
    # (batch_size, seq_len)
    batch_size, seq_len = 8, 10
    model_path = config.get("pretrained").get("bert")

    input_ids = torch.randint(100, 1000, (batch_size, seq_len)).tolist()

    pretrained_processor = PretrainedProcessor()
    pretrained_model = Pretrained("bert", model_path, True)

    outputs = pretrained_model(pretrained_processor(input_ids))

    for output in outputs:
        assert len(output) == 2
        assert output[0].shape == torch.Size(
            [batch_size, seq_len, pretrained_model.node.model.config.hidden_size])
        assert output[1].shape == torch.Size(
            [batch_size, pretrained_model.node.model.config.hidden_size])


def test_bert_pretrained_with_pipe():
    batch_size, seq_len = 8, 10
    model_path = config.get("pretrained").get("bert")

    input_ids = torch.randint(100, 1000, (batch_size, seq_len)).tolist()

    pipe = (
        PretrainedProcessor(name="pretrained") >>
        Pretrained("bert", model_path, True)
    )
    outputs = pipe.run(input_ids)

    for output in outputs:
        assert len(output) == 2
        assert output[0].shape == torch.Size(
            [batch_size, seq_len, 768])
        assert output[1].shape == torch.Size(
            [batch_size, 768])


def test_bert_pretrained_with_func_pipe():
    batch_size, seq_len = 8, 10
    model_path = config.get("pretrained").get("bert")

    input_ids = torch.randint(100, 1000, (batch_size, seq_len)).tolist()
    
    pretrained_processor = PretrainedProcessor(name="pretrained")
    pretrained_model = Pretrained("bert", model_path, True)

    pipe = N(pretrained_processor) >> N(pretrained_model)

    outputs = pipe(input_ids)

    for output in outputs:
        assert len(output) == 2
        assert output[0].shape == torch.Size(
            [batch_size, seq_len, pretrained_model.node.model.config.hidden_size])
        assert output[1].shape == torch.Size(
            [batch_size, pretrained_model.node.model.config.hidden_size])
