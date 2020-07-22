import torch
from torch import Tensor

from hnlp.task.classification import BertFcClassifier


import pnlp


config = pnlp.read_yaml("tests/config.yaml")


def test_bert_fc_classifier_training():
    model = BertFcClassifier(
        pretrained_path=config.get("pretrained").get("bert"),
        is_training=True
    )
    batch_size, seq_len = 8, 10
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    logits = model(input_ids)
    assert logits.shape == torch.Size([batch_size, model.num_labels])
    assert logits.requires_grad == True


def test_bert_fc_classifier_not_trainint():
    model = BertFcClassifier(
        pretrained_path=config.get("pretrained").get("bert"),
        is_training=False
    )
    batch_size, seq_len = 8, 10
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    logits = model(input_ids)
    assert logits.shape == torch.Size([batch_size, model.num_labels])
    assert logits.requires_grad == False
