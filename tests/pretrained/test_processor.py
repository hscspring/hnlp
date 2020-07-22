import torch
from hnlp.pretrained.processor import PretrainedProcessor


def test_bert_pretrained_processor():
    pp = PretrainedProcessor("bert")
    batch_size, seq_len = 2, 4
    batch = [
        [1, 2, 3, 4],
        [0, 1, 2, 3]
    ]

    outputs = pp(batch)

    assert len(outputs) == 4
    assert outputs["input_ids"].shape == torch.Size([batch_size, seq_len])
    assert outputs["attention_mask"].shape == torch.Size([batch_size, seq_len])
    assert outputs["token_type_ids"].shape == torch.Size([batch_size, seq_len])
    assert outputs["position_ids"].shape == torch.Size([batch_size, seq_len])
