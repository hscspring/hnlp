import os
import logging
from pathlib import Path
from addict import Dict as ADict
from typing import TypeVar, List, Dict
import pnlp
import numpy as np
import torch
from torch import Tensor


ARCH = os.environ.get("ARCH") or "tf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModelInputType = TypeVar("ModelInputType", np.array, Tensor, Dict[str, Tensor])
ModelLabelType = TypeVar("ModelLabelType", List[str], List[int], Tensor)


logger = logging.getLogger("hnlp")
logging.basicConfig(level=logging.INFO)

home = Path.home() / ".hnlp"
model_home = home / "model"
data_home = home / "dataset"
pnlp.check_dir(model_home)
pnlp.check_dir(data_home)


root = Path(os.path.abspath(__file__)).parent

model_config = ADict(
    {
        "fasttext_word2vec": pnlp.read_json(root / "fasttext_word2vec.json"),
        "cnn": pnlp.read_json(root / "cnn.json"),
        "gru": pnlp.read_json(root / "gru.json"),
    }
)


SpeToken = ADict(
    {
        "pad": "[PAD]",
        "unk": "[UNK]",
        "cls": "[CLS]",
        "sep": "[SEP]",
        "mask": "[MASK]",
        "s": "<S>",
        "t": "<T>",
        "unused": "[unused{}]",
    }
)


def check_name(identity: str, name: str):
    if identity == "pretrained_model":
        return name in ["fasttext", "bert"]
