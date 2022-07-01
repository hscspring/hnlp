import tensorflow.keras.backend as K
import tensorflow.keras as tfk
import tensorflow as tf

from typing import List
from pathlib import Path
import pnlp
from hnlp import DataManager, Corpus, Tokenizer
from hnlp.trainer import Trainer, Loss, MetricStep
from transformers import TFAutoModel, AutoConfig
from pnlp import MagicDict

from hnlp import cnn

data_root = Path("/home/hsc/ner/intent/")
bert_root = Path("/home/hsc/ner/model/chinese_wwm_ext_L-12_H-768_A-12/")

labels = pnlp.read_lines(data_root / "labels.txt")
label_map = dict(zip(labels, range(len(labels))))


config = MagicDict({

    "vocab_size": 21128,
    "embed_size": 300,

    "hidden_size": 512,
    "dropout": 0.1,

    "filter_sizes": "2,3,4",
    "num_filters": 128,

    "num_heads": 12,
    "key_dim": 64,

    "initializer_range": 0.02,
    "regularizers_lambda": 0.01,

    "max_seq_len": 50,
    "num_labels": len(labels),
    "learning_rate": 2e-5,

    "batch_size": 128,
    "epochs": 20,
    "use_pretrained_bert": True,
    "pretrained_bert_path": bert_root.as_posix(),

    "out_path": "./output/",
})


class Model(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.label_size = config.num_labels
        self.dropout_rate = config.dropout

        bert_config = AutoConfig.from_pretrained(bert_root / "config.json")
        self.bert = TFAutoModel.from_config(bert_config)
        self.dense = tfk.layers.Dense(
            units=self.label_size,
            activation="softmax"
        )
        self.dropout = tfk.layers.Dropout(self.dropout_rate)
        # self.embed = tfk.layers.Embedding(config.vocab_size, config.embed_size)

    # @tf.function
    def call(self, x, y=None, training=None):
        z = self.bert(x)
        # z = self.embed(x)
        z = self.dropout(z.pooler_output, training)
        # z = cnn(self.config, z)
        output = self.dense(z)
        return output


def build_model(config, module):
    inp = tfk.layers.Input(shape=(config.max_seq_len, ),
                           dtype="int32", name="input")
    z = tfk.layers.Embedding(config.vocab_size, config.embed_size)(inp)
    z = module(config, z)
    z = tf.keras.layers.Dropout(rate=config.dropout)(z)
    out = tfk.layers.Dense(
        units=config.num_labels,
        activation="softmax",
        kernel_initializer="glorot_normal",
        bias_initializer=tfk.initializers.constant(config.initializer_range),
        kernel_regularizer=tfk.regularizers.l2(config.regularizers_lambda),
        bias_regularizer=tfk.regularizers.l2(config.regularizers_lambda),
        name="dense"
    )(z)
    md = tfk.Model(inputs=[inp], outputs=[out])
    return md


cs = Corpus(
    name="labeled",
    keys=("text", "label"),
    label_map=label_map,
)
tk = Tokenizer(
    name="bert"
)
dm_train = DataManager(
    name="random",
    batch_size=config.batch_size,
    max_seq_len=config.max_seq_len
)
dm_test = DataManager(
    name="sequence",
    batch_size=config.batch_size,
    max_seq_len=config.max_seq_len,
    split_val=False
)


train_ds, val_ds = (cs >> tk >> dm_train).run(data_root/"train.txt")
test_ds = (cs >> tk >> dm_test).run(data_root/"test.txt")

for x, y in test_ds:
    print(x, y)

# model = build_model(config, cnn)
model = Model(config)
trainer = Trainer(config)


trainer.train(model, Loss.crossentropy_loss,
              MetricStep.sequence_classification, train_ds, val_ds)

trainer.train(model, train_ds, val_ds)
acc, loss, report, confusion = trainer.evaluate(
    model, Loss.crossentropy_loss, MetricStep.sequence_classification, test_ds, True)
print(report)
