# from model import TokenClassificationSoftmaxModel as Model
import tensorflow as tf
from model import TokenClassificationCrfModel as Model
from hnlp import Corpus, Tokenizer, DataManager, Trainer, Node
from hnlp.trainer import Loss, MetricStep
import pnlp
from pnlp import MagicDict
from pathlib import Path

data_root = Path("/home/hsc/ner/v0.2.0/")
label_list = pnlp.read_json(data_root / "labels.json")
label_list.insert(0, "PAD")
i = 0
label_map = {}
for it in label_list:
    label_map[it] = i
    i += 1
print(label_map)


class CustomPreprocessor(Node):
    def __init__(self):
        super().__init__()
        self.node = lambda x: " ".join(list(x)) if type(x) == str else x


config = MagicDict({
    "vocab_size": 21128,
    "embed_size": 200,
    "hidden_size": 512,
    "dropout": 0.5,
    "max_seq_len": 200,
    "num_labels": len(label_list),
    "label_list": label_list,

    "epochs": 20,
    "learning_rate": 1e-3,
    "use_decay": False,
    "valid_epochs": 0.5,
    "batch_size": 32,
    "out_path": "./crf_output/"
})


cs = Corpus(
    name="labeled",
    keys=("text", "label"),
    shuffle=False,
    label_map=label_map,
    add_special_label=True,
)
pp = CustomPreprocessor()
tk = Tokenizer(
    name="bert",
    max_seq_len=config.max_seq_len
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


train_ds, val_ds = (cs >> pp >> tk >> dm_train).run(data_root/"train.txt")
test_ds = (cs >> pp >> tk >> dm_test).run(data_root/"test.txt")

for x, y in (cs >> pp >> tk >> dm_test).run(data_root/"test.txt"):
    print(x[0])
    print(y[0])
    break

model = Model(config)
trainer = Trainer(config)
trainer.train(
    model,
    Loss.mean_loss_crf,
    MetricStep.token_classification_crf,
    train_ds, val_ds
)
acc, loss, report, confusion = trainer.evaluate(
    model,
    Loss.mean_loss_crf,
    MetricStep.token_classification_crf,
    test_ds, True
)
print(report)


# optimizer = tf.keras.optimizers.Adam(1e-3)
# for _, (x, y_true) in enumerate(train_ds):
#     with tf.GradientTape() as tape:
#         output = model(x, y_true, training=True)
#         loss = Loss.mean_loss_crf(output, y_true)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     print(model.trainable_variables)