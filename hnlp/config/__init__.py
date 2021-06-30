import logging
from pathlib import Path
from typing import TypeVar, List, Tuple, Dict
import os
import pnlp
import torch
from torch import Tensor

logger = logging.getLogger("hnlp")
logging.basicConfig(level=logging.INFO)

home = Path("~/.hnlp")
default_model_home = home / "model"
default_data_home = home / "dataset"
pnlp.check_dir(default_model_home)
pnlp.check_dir(default_data_home)


root = Path(os.path.abspath(__file__)).parent
default_fasttext_word2vec_config = pnlp.read_json(
    root / "pretrained/fasttext/word2vec.json"
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModelInputType = TypeVar(
    "ModelInputType", Tensor, List[List[int]], Tuple[Tensor], Dict[str, Tensor]
)
ModelLabelType = TypeVar("ModelLabelType", List[str], List[int], Tensor)
