from hnlp.layer.embeddings_tf import Embeddings
from hnlp.sampler.sampler_tf import gen_input
from pnlp import MagicDict


def test_work():
    x, y = gen_input(3, 60, 1, 2)
    config = MagicDict()
    embed = Embeddings(config)
    out = embed(x)
    assert out.shape.as_list() == [3, 60, 300]
