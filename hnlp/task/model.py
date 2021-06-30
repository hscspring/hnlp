from dataclasses import dataclass
from argparse import Namespace

import time

from hnlp.node import Node
from hnlp.register import Register
from hnlp.utils import convert_model_input, convert_input, convert_label
from hnlp.config import device, ModelInputType, logger
from hnlp.task.trainer import Trainer


@dataclass
class Model(Node):

    """
    When is_training is True, model_path is actually the pretrained_model_path;
    When is_training is False, we are doing inference, so model_path is the trained_model_path.
    """
    name: str
    is_training: bool = False
    args: Namespace = Namespace()

    def __post_init__(self):
        super().__init__()
        self.identity = "task"
        self.batch = True
        TaskModel = Register.get(self.name)
        if not TaskModel:
            raise NotImplementedError
        task_model = TaskModel(self.is_training).to(device)
        self.node = task_model
        if self.is_training:
            self.trainer = Trainer(self.args, task_model)
        # if self.is_training:
        #     self.node = task_model
        #     self.trainer = Trainer(self.args, task_model)
        # else:
        #     state_dict_file = os.path.join(
        #         [self.pretrained_path, "pytorch_model.bin"])
        #     state_dict = torch.load(state_dict_file)
        #     self.node = task_model.load_state_dict(state_dict)

    def fit(self, train_dataloader):
        for epoch in range(self.trainer.n_epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_func(train_dataloader)
            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            logger.info('Epoch: %d' % (epoch + 1),
                        " | time in %d minutes, %d seconds" % (mins, secs))
            logger.info(f'\t\
                Loss: {sum(train_loss)/len(train_dataloader.dataset): .4f}(train)\t\
                Acc: {sum(train_acc)/len(train_dataloader.dataset) * 100: .1f} % (train)')

    def train_func(self, dataloader):
        history = []
        accuracy = []
        for batch in dataloader:
            inputs, labels = batch
            inputs = convert_input(inputs)
            labels = convert_label(labels)
            logits = self.node(inputs)
            loss = self.trainer.criterion(logits, labels)
            history.append(loss.item())
            loss.backward()
            self.trainer.optimizer.step()
            acc = (logits.argmax(1) == labels).sum().item()
            accuracy.append(acc)
        self.trainer.scheduler.step()
        return history, accuracy

    @convert_model_input
    def predict(self, inputs: ModelInputType):
        logits = self.node(inputs)
        return logits
