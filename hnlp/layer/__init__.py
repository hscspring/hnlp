from hnlp.config import ARCH


if ARCH == "tf":
    from hnlp.layer.embeddings_tf import Embeddings
    from hnlp.layer.attn_tf import InteractiveSelfAttention
else:
    raise NotImplementedError
