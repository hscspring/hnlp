import time
from typing import Callable, List, Union, Tuple, Any
import tensorflow as tf
import tensorflow.keras as tfk


import numpy as np
from sklearn import metrics
from pnlp import MagicDict
from transformers.optimization_tf import WarmUp


from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from hnlp.dataset.datamanager_tf import DataLoader
from tensor_annotations.axes import Batch, Time
import tensor_annotations.tensorflow as ttf

from hnlp.config import logger
from hnlp.utils import unfold


class Trainer:

    def __init__(self, config: MagicDict):
        self.config = config

        self.use_tf_function = config.use_tf_function or False

        # Training
        self.epochs = config.epochs or 100
        self.batch_size = config.batch_size or 32
        self.learning_rate = config.learning_rate or 1e-3
        self.use_decay = config.use_decay or True
        self.use_warmup = config.use_warmup or False
        self.early_stop_times = config.early_stop_times or 3  # 几个 Epoch 没有提升就提前终止
        self.valid_times = config.valid_times or 0.2  # Epoch 内多少比例 Step 进行一次验证
        self.early_stop_steps = config.early_stop_steps or None
        self.valid_steps = config.valid_steps or None

        self.ckpt_path = config.ckpt_path or "./output/ckpt/"
        self.logs_path = config.logs_path or "./output/logs/"
        self.save_path = config.save_path or "./output/save/"

        self.label_list = config.label_list or None

    @staticmethod
    def get_acc(
        y_preds: List[Union[int, List[int]]],
        y_trues: List[Union[int, List[int]]],
    ) -> float:
        acc = 0
        for i, (y_pred, y_true) in enumerate(zip(y_preds, y_trues), start=1):
            pa = np.array(y_pred, dtype=np.int16)
            ta = np.array(y_true, dtype=np.int16)
            acc += np.average(pa == ta)
        return acc / i

    def get_lr_schedule(
        self,
        data_size: int
    ) -> Union[float, LearningRateSchedule]:
        steps_per_epoch = np.ceil(data_size / self.batch_size).astype(np.int32)

        info = f"Epochs: {self.epochs}, Steps per epoch: {steps_per_epoch}"
        logger.info(info)

        lr = self.learning_rate
        if self.use_decay:
            lr = tfk.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=steps_per_epoch,  # 每隔 decay_step decay 一次
                decay_rate=0.96,
                staircase=True,
            )
            if self.use_warmup:
                lr = WarmUp(
                    initial_learning_rate=self.learning_rate,
                    scheduler=lr,
                    warmup_step=steps_per_epoch // 2)
        return lr

    def evaluate(
        self,
        model: Callable,
        loss_fn: Callable,
        metric_step: Callable,
        dataset: DataLoader,
        print_report: bool = False
    ):

        total_loss = 0.0
        steps = 0
        y_pred_all = []
        y_true_all = []
        for step, (*inputs, labels) in enumerate(dataset, start=1):
            if len(inputs) == 1:
                inputs = (inputs[0], )
            else:
                inputs = tuple(inputs)
            loss, output = self.test_step(model, loss_fn, *inputs, labels)
            total_loss += loss
            y_preds, y_trues = metric_step(model, output, labels)

            y_pred_all.append(y_preds)
            y_true_all.append(y_trues)

            steps += 1

        ps = unfold(y_pred_all)
        ts = unfold(y_true_all)
        acc = Trainer.get_acc(ps, ts)
        loss = total_loss / steps

        if print_report:
            report = metrics.classification_report(
                ts, ps, target_names=self.label_list, digits=4
            )
            confusion = metrics.confusion_matrix(ts, ps)
            tf.print(report)
            tf.print(confusion)
            tf.print(f"TestLoss: {loss:.4f}  |  TestMse/Acc: {acc:.4f}\n")
        else:
            return acc, loss

    def _train_step(
        self,
        model: Callable,
        optimizer: Optimizer,
        loss_fn: Callable,
        inp: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time],
    ) -> Tuple[float, Any]:
        with tf.GradientTape() as tape:
            output = model(inp, y_true, training=True)
            loss = loss_fn(output)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(
            grads_and_vars=zip(
                grads, model.trainable_weights))
        return loss, output

    def train_step(
        self,
        model: Callable,
        optimizer: Optimizer,
        loss_fn: Callable,
        inp: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time],
    ) -> Tuple[float, Any]:
        if self.use_tf_function:
            func = tf.function(self._train_step)
        else:
            func = self._train_step
        return func(model, optimizer, loss_fn, inp, y_true)

    def _test_step(
        self,
        model: Callable,
        loss_fn: Callable,
        inp: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time]
    ) -> Tuple[float, Any]:
        output = model(inp, y_true, training=False)
        loss = loss_fn(output)
        return loss, output

    def test_step(
        self,
        model: Callable,
        loss_fn: Callable,
        inp: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time]
    ) -> Tuple[float, Any]:
        if self.use_tf_function:
            func = tf.function(self._test_step)
        else:
            func = self._test_step
        return func(model, loss_fn, inp, y_true)

    def train(
        self,
        model: Callable,
        optimizer: Optimizer,
        loss_fn: Callable,
        metric_step: Callable,
        train_dataset: DataLoader,
        val_dataset: DataLoader
    ):

        start_time = time.perf_counter()
        checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            optimizer=optimizer,
            model=model
        )
        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self.ckpt_path,
            max_to_keep=3
        )

        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logger.info(f"Restored from {manager.latest_checkpoint}")
        else:
            logger.info("Initializing from scratch.")

        summary_writer = tf.summary.create_file_writer(self.logs_path)
        val_best_loss = float("inf")
        last_improve = 0
        flag = False

        data_size = len(train_dataset)
        # batches
        epoch_steps = np.ceil(data_size / self.batch_size).astype(np.int32)
        # 连续3个epoch没有提升
        early_stop_steps = self.early_stop_times * epoch_steps
        # 1/5个epoch验证一次
        valid_steps = self.valid_steps or int(self.valid_times * epoch_steps)
        valid_steps = max(1, valid_steps)

        tf.print(
            f"Epoch steps: {epoch_steps}, Valid steps: {valid_steps}, Early stop steps: {early_stop_steps}"
        )

        for epoch in range(self.epochs):
            tf.print(f"\nEpoch {epoch+1}/{self.epochs}")
            train_loss = 0.0
            train_acc = 0.0
            step = 0
            for step, (*inputs, labels) in enumerate(train_dataset, start=1):
                if len(inputs) == 1:
                    inputs = (inputs[0], )
                else:
                    inputs = tuple(inputs)
                loss, output = self.train_step(
                    model, optimizer, loss_fn, *inputs, labels
                )
                train_loss += loss.numpy()

                y_preds, y_trues = metric_step(model, output, labels)
                acc = Trainer.get_acc(y_preds, y_trues)
                train_acc += acc

                step += 1
                total_step = int(checkpoint.step)

                if total_step > valid_steps and total_step % valid_steps == 0:
                    val_acc, val_loss = self.evaluate(
                        model, loss_fn, metric_step, val_dataset
                    )
                    if val_loss < val_best_loss:
                        val_best_loss = val_loss
                        last_improve = total_step
                        path = manager.save(checkpoint_number=total_step)
                        tf.print(f"Model saved to {path}")

                    msg = f"Total Step: {total_step},  Step(Batch): {step},  "
                    msg += f"LearningRate: {optimizer.learning_rate(total_step).numpy():.6f},  \n"
                    msg += f"TrainLoss: {loss.numpy():.4f},  TrainAcc:{acc:.4f},  "
                    msg += f"ValidLoss: {val_loss:.4f},  ValidAcc: {val_acc:.4f}\n"
                    msg += "- " * 30
                    tf.print(msg)

                    with summary_writer.as_default():
                        tf.summary.scalar(
                            "train_loss", loss.numpy(), step=total_step)
                        tf.summary.scalar(
                            "val_loss", val_loss, step=total_step)
                        tf.summary.scalar("train_acc", acc, step=total_step)
                        tf.summary.scalar("val_acc", val_acc, step=total_step)

                checkpoint.step.assign_add(1)

                if total_step - \
                        last_improve > (self.early_stop_steps or early_stop_steps):
                    tf.print("Early stop for no improvements...")
                    flag = True
                    break

            secs = int(time.perf_counter() - start_time)
            mins = secs / 60
            secs = secs % 60

            val_acc, val_loss = self.evaluate(
                model, loss_fn, metric_step, val_dataset
            )

            tf.print(
                f"Epoch: {epoch+1} | time in {mins:.1f} minutes, {secs} seconds")
            tf.print(
                f"\tTrainLoss: {train_loss/step:.4f}  |  TrainMse/Acc: {train_acc/step:.4f}")
            tf.print(
                f"\tValidLoss: {val_loss:.4f}  |  ValidMse/Acc: {val_acc:.4f}")

            if flag:
                break
