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

logger = logging.getLogger("hnlp")
logging.basicConfig(level=logging.INFO)

home = Path.home() / ".hnlp"
default_model_home = home / "model"
default_data_home = home / "dataset"
pnlp.check_dir(default_model_home)
pnlp.check_dir(default_data_home)


root = Path(os.path.abspath(__file__)).parent
default_fasttext_word2vec_config = pnlp.read_json(root / "fasttext_word2vec.json")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModelInputType = TypeVar("ModelInputType", np.array, Tensor, Dict[str, Tensor])
ModelLabelType = TypeVar("ModelLabelType", List[str], List[int], Tensor)


def check_name(identity: str, name: str):
    if identity == "pretrained_model":
        return name in ["fasttext", "bert"]


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
