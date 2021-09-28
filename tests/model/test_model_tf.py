from hnlp.model.model_tf import text_cnn, text_gru
from hnlp.config import model_config
from hnlp.sampler import gen_hidden


def test_text_cnn_work():
    embed = gen_hidden(32, 512, 300)
    z = text_cnn(model_config.text_cnn.model, embed)
    assert z.shape.as_list() == [32, 384]


def test_text_gru_work():
    embed = gen_hidden(32, 512, 300)
    z = text_gru(model_config.text_gru.model, embed, mask=None)
    assert z.shape.as_list() == [32, 256]
