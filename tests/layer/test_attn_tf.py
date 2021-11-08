import pytest
from hnlp.layer.attn_tf import InteractiveSelfAttention
from hnlp.sampler.sampler_tf import gen_hidden
from pnlp import MagicDict


@pytest.mark.parametrize("attn_type", ["general", "dot"])
def test_work(attn_type):
    x = gen_hidden(3, 20, 32)
    config = MagicDict()
    config.attention_type = attn_type
    attn = InteractiveSelfAttention(config)
    out = attn(x)
    assert out.shape.as_list() == [3, 20, 32]
