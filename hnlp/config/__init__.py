import os
import logging
from pathlib import Path
import pnlp
from pnlp import MagicDict


ARCH = os.environ.get("ARCH") or "tf"


logger = logging.getLogger("hnlp")
logging.basicConfig(level=logging.INFO)

home = Path.home() / ".hnlp"
model_home = home / "model"
data_home = home / "dataset"
pnlp.check_dir(model_home)
pnlp.check_dir(data_home)


root = Path(os.path.abspath(__file__)).parent

model_root = root / "model"
vocab_root = root / "vocab"


default_config = MagicDict(
    {
        "fasttext_word2vec": pnlp.read_json(model_root / "fasttext_word2vec.json"),
        "cnn": pnlp.read_json(model_root / "cnn.json"),
        "gru": pnlp.read_json(model_root / "gru.json"),
        "vocab_file": vocab_root / "bert/vocab.txt",
        "train_tf": pnlp.read_json(model_root / "train_tf.json"),
    }
)


SpeToken = MagicDict(
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
