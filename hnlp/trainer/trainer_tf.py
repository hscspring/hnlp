import os.path as osp
import time
from tkinter import N
from typing import Callable, List, Union, Tuple, Any
import tensorflow as tf
import tensorflow.keras as tfk

import numpy as np
from sklearn import metrics
from pnlp import MagicDict
from transformers.optimization_tf import WarmUp

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from hnlp.dataset.datamanager_tf import DataLoader
from tensor_annotations.axes import Batch, Time
import tensor_annotations.tensorflow as ttf

from hnlp.utils import unfold
from hnlp.config import default_config



class Trainer:

    def __init__(self, config: dict = {}):
        self._config = MagicDict({**default_config.train_df, **config})
        for key, val in self._config.items():
            setattr(self, key, val)

        self.train_step_func = None
        self.test_step_func = None

    @property
    def config(self):
        return self._config
    

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
        decay_steps = self.decay_epochs * steps_per_epoch

        info = f"Epochs: {self.epochs}, Steps per epoch: {steps_per_epoch}"
        tf.print(info)

        lr = self.learning_rate
        if self.use_decay:
            lr = tfk.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=decay_steps,  # 每隔 decay_step decay 一次
                decay_rate=0.96,
                staircase=True,
            )
            if self.use_warmup:
                lr = WarmUp(
                    initial_learning_rate=self.learning_rate,
                    scheduler=lr,
                    warmup_step=steps_per_epoch // 2)
        return lr

    def get_optimizer(self, optimizer: str, data_size: int):
        schedule = self.get_lr_schedule(data_size)
        optimizer_func = getattr(tfk.optimizers, optimizer)(
            learning_rate=schedule)
        return optimizer_func

    def _train_step(
        self,
        model: Callable,
        loss_fn: Callable,
        inp: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time],
    ) -> Tuple[float, Any]:
        with tf.GradientTape() as tape:
            output = model(inp, y_true, training=True)
            loss = loss_fn(output, y_true)
        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars_mult = []

        for grad, var in zip(grads, model.trainable_variables):
            if grad is None:
                continue
            if hasattr(model, "get_learning_rate"):
                lr_dct = model.get_learning_rate()
                key = var.name
                if key not in lr_dct:
                    key = key.split(":")[0]
                var_lr = lr_dct.get(key)
                if var_lr:
                    grad *= var_lr
            grads_and_vars_mult.append((grad, var))

        self.optimizer.apply_gradients(
            grads_and_vars=grads_and_vars_mult)
        return loss, output

    def _test_step(
        self,
        model: Callable,
        loss_fn: Callable,
        inp: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time]
    ) -> Tuple[float, Any]:
        output = model(inp, y_true, training=False)
        loss = loss_fn(output, y_true)
        return loss, output

    def train_step(
        self,
        model: Callable,
        loss_fn: Callable,
        *inp: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time],
    ) -> Tuple[float, Any]:
        if self.use_tf_function:
            func = tf.function(self._train_step)
        else:
            func = self._train_step
        if len(inp) == 1:
            return func(model, loss_fn, inp, y_true)
        else:
            return func(model, loss_fn, inp1, inp2, y_true)

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
            # ps may contain 0, change them to 1
            o_idx = self.label_list.index("O")
            for i, v in enumerate(ps):
                if v == 0:
                    ps[i] = o_idx
            # drop `PAD` label
            self.label_list.pop(0)
            report = metrics.classification_report(
                ts, ps, target_names=self.label_list, digits=4
            )
            confusion = metrics.confusion_matrix(ts, ps)
            tf.print(f"TestLoss: {loss:.4f}  |  TestMse/Acc: {acc:.4f}\n")
            return acc, loss, report, confusion
        else:
            return acc, loss

    def train(
        self,
        model: Callable,
        loss_fn: Callable,
        metric_step: Callable,
        train_dataset: DataLoader,
        val_dataset: DataLoader
    ):

        start_time = time.perf_counter()
        data_size = len(train_dataset)
        self.optimizer = self.get_optimizer(self.optimizer_str, data_size)

        checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            optimizer=self.optimizer,
            model=model
        )
        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=osp.join(self.out_path, "ckpt"),
            max_to_keep=3
        )

        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            tf.print(f"Restored from {manager.latest_checkpoint}")
        else:
            tf.print("Initializing from scratch.")

        summary_writer = tf.summary.create_file_writer(osp.join(self.out_path, "logs"))
        val_best_loss = float("inf")
        last_improve = 0
        flag = False

        # batches
        epoch_steps = np.ceil(data_size / self.batch_size).astype(np.int32)
        # 连续3个epoch没有提升
        early_stop_steps = self.early_stop_epochs * epoch_steps
        # 1/5个epoch验证一次
        valid_steps = self.valid_steps or int(self.valid_epochs * epoch_steps)
        valid_steps = max(1, valid_steps)

        msg = f"Epoch steps: {epoch_steps}, "
        msg += f"Valid steps: {valid_steps}, "
        msg += f"Early stop steps: {early_stop_steps}"
        tf.print(msg)

        if valid_steps >= epoch_steps:
            msg = "Valid steps is bigger than epoch steps."
            msg += "Use the epoch ealuation, instead of evaluation in the epoch."
            tf.print(msg)

        for epoch in range(1, self.epochs + 1):
            tf.print(f"\nEpoch {epoch}/{self.epochs}")
            train_loss = 0.0
            train_acc = 0.0
            step = 0
            for step, (*inputs, labels) in enumerate(train_dataset, start=1):
                if len(inputs) == 1:
                    inputs = (inputs[0], )
                else:
                    inputs = tuple(inputs)
                loss, output = self.train_step(model, loss_fn, *inputs, labels)
                train_loss += loss.numpy()

                # Might need some model parameters to calculate y_preds
                y_preds, y_trues = metric_step(model, output, labels)
                acc = Trainer.get_acc(y_preds, y_trues)
                train_acc += acc

                step += 1
                total_step = int(checkpoint.step)

                if (
                    total_step > valid_steps and
                    total_step % valid_steps == 0 and
                    # if valid steps bigger than epoch steps, use the epoch evaluation
                    valid_steps < epoch_steps
                ):
                    val_acc, val_loss = self.evaluate(
                        model, loss_fn, metric_step, val_dataset)
                    if val_loss < val_best_loss:
                        val_best_loss = val_loss
                        last_improve = total_step
                        path = manager.save(checkpoint_number=total_step)
                        tf.print(f"Model saved to {path}")

                    msg = f"Total Step: {total_step},  Step(Batch): {step},  "
                    if self.use_decay:
                        msg += f"LearningRate: {self.optimizer.learning_rate(total_step).numpy():.6f}, \n"
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

            tf.print("step", step)
            secs = int(time.perf_counter() - start_time)
            mins = secs / 60
            secs = secs % 60

            val_acc, val_loss = self.evaluate(
                model, loss_fn, metric_step, val_dataset)

            msg = f"Epoch: {epoch} | time in {mins:.1f} minutes, {secs} seconds \n"
            msg += f"\tEpoch TrainLoss: {train_loss/step:.4f}  |  Epoch TrainMse/Acc: {train_acc/step:.4f} \n"
            msg += f"\tEpoch ValidLoss: {val_loss:.4f}  |  Epoch ValidMse/Acc: {val_acc:.4f}"
            tf.print(msg)

            if flag:
                break
        checkpoint.restore(manager.latest_checkpoint)
        tf.saved_model.save(model, osp.join(self.out_path, "save"))
