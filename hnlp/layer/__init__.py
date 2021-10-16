from hnlp.config import ARCH


if ARCH == "tf":
    from hnlp.layer.embeddings_tf import Embeddings
else:
    raise NotImplementedError
