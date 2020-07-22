import torch
from torch import Tensor
from hnlp.pretrained.model import Pretrained
import pnlp

config = pnlp.read_yaml("tests/config.yaml")


def test_bert_pretrained():
    # (batch_size, seq_len)
    batch_size, seq_len = 8, 10
    model_path = config.get("pretrained").get("bert")

    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    pretrain = Pretrained("bert", model_path)
    outputs = pretrain(inputs=input_ids)

    assert len(outputs) == 2
    assert outputs[0].shape == torch.Size(
        [batch_size, seq_len, pretrain.node.model.config.hidden_size])
    assert outputs[1].shape == torch.Size(
        [batch_size, pretrain.node.model.config.hidden_size])
    assert outputs[0].requires_grad == False
    assert outputs[1].requires_grad == False


def test_bert_pretrained_training():
    # (batch_size, seq_len)
    batch_size, seq_len = 8, 10
    model_path = config.get("pretrained").get("bert")

    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    pretrain = Pretrained("bert", model_path, is_training=True)
    outputs = pretrain(inputs=input_ids)

    assert len(outputs) == 2
    assert outputs[0].shape == torch.Size(
        [batch_size, seq_len, pretrain.node.model.config.hidden_size])
    assert outputs[1].shape == torch.Size(
        [batch_size, pretrain.node.model.config.hidden_size])
    assert outputs[0].requires_grad == True
    assert outputs[1].requires_grad == True


def test_bert_pretrained_batch_list():
    # (batch_size, seq_len)
    batch_size, seq_len = 2, 4
    model_path = config.get("pretrained").get("bert")

    batch = [
        [1, 2, 3, 4],
        [0, 1, 2, 3]
    ]
    pretrain = Pretrained("bert", model_path)
    outputs = pretrain(inputs=batch)

    assert len(outputs) == 2
    assert outputs[0].shape == torch.Size(
        [batch_size, seq_len, pretrain.node.model.config.hidden_size])
    assert outputs[1].shape == torch.Size(
        [batch_size, pretrain.node.model.config.hidden_size])


def test_bert_pretrained_return_all_hidden_states():
    batch_size, seq_len = 8, 10
    model_path = config.get("pretrained").get("bert")

    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    pretrain = Pretrained("bert", model_path,
                          is_training=True, output_hidden_states=True)
    outputs = pretrain(input_ids)

    assert len(outputs) == 3
    assert len(outputs[2]) == pretrain.node.model.config.num_hidden_layers + 1
    assert outputs[2][0].shape == torch.Size(
        [batch_size, seq_len, pretrain.node.model.config.hidden_size])


def test_bert_pretrained_return_all_attentions():
    batch_size, seq_len = 8, 10
    model_path = config.get("pretrained").get("bert")

    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    pretrain = Pretrained("bert", model_path, output_attentions=True)
    outputs = pretrain(input_ids)

    assert len(outputs) == 3
    assert len(outputs[2]) == pretrain.node.model.config.num_hidden_layers
    assert outputs[2][0].shape == torch.Size(
        [batch_size,
         pretrain.node.model.config.num_attention_heads,
         seq_len,
         seq_len]
    )
