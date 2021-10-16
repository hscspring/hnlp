from hnlp.dataset.corpus import Corpus
from hnlp.dataset.preprocessor import Preprocessor
from hnlp.dataset.tokenizer import Tokenizer
from hnlp.dataset.dataset import MapStyleDataset

from hnlp.config import ARCH


if ARCH == "tf":
    from hnlp.dataset.datamanager_tf import DataManager
elif ARCH == "pt":
    from hnlp.dataset.datamanager_pt import DataManager
