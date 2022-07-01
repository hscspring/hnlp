from pathlib import Path
import json
import pnlp
from hnlp import DataManager, Corpus, Node
from transformers import TFAutoModel, AutoTokenizer, AutoConfig
from pnlp import MagicDict

from hnlp.dataset.dataset import MapStyleDataset

data_root = Path("/home/hsc/ner/v0.2.0/")
bert_root = Path("/home/hsc/ner/model/chinese_wwm_ext_L-12_H-768_A-12/")

label_list = pnlp.read_json(data_root / "labels.json")
label_list.insert(0, "UNK")
i = 0
label_map = {}
for it in label_list:
    label_map[it] = i
    i += 1


class CustomPreprocessor(Node):
    def __init__(self):
        super().__init__()
        self.node = lambda x: " ".join(list(x)) if type(x) == str else x


config = MagicDict({
    "vocab_size": 21128,
    "dropout": 0.1,
    "max_seq_len": 512,
    "num_labels": len(label_list),
    "label_list": label_list,

    "learning_rate": 1e-3,
    "valid_epochs": 0.5,
    "batch_size": 8,
    "out_path": "./output/"
})


tk = AutoTokenizer.from_pretrained(bert_root.as_posix())
bert_config = AutoConfig.from_pretrained(bert_root / "bert_config.json")
model = TFAutoModel.from_config(bert_config)


def collate_fn(batch, max_seq_len=512, dynamic_length=False):
    labels = [b[1] for b in batch]
    slist = [b[0] for b in batch]

    padded = MapStyleDataset.padding_tokens(
        labels, max_seq_len, dynamic_length, 0)
    return (slist, padded)


pipe = (
    Corpus(
        name="labeled",
        keys=("text", "label"),
        shuffle=False,
        label_map=label_map,
    ) >>
    CustomPreprocessor() >>
    DataManager(
        name="sequence",
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        collate_fn=collate_fn,
        return_type="list",
    )
)


for (slist, labels) in pipe.run(data_root / "dev.txt"):
    inp = tk(
        slist,
        padding="max_length",
        max_length=30,
        truncation=True, return_tensors="np")
    print(inp["input_ids"])
    break
