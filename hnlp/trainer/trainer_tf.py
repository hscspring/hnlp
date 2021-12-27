import time
import tensorflow as tf
from pnlp import MagicDict


class Trainer:

    def __init__(self, config: MagicDict):
        self.config = config

    def train(
        self,
        model,
        optimizer,
        loss_fn,
        train_dataset,
        valid_dataset,
        train_metric,
        valid_metric,
    ):
        start_time = time.time()
        for epoch in range(self.config.n_epochs):
            train_acc, train_loss = self.run_epoch(
                self.config,
                model,
                optimizer,
                loss_fn,
                train_dataset,
                train_metric,
                True,
            )

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            val_acc, val_loss = self.run_epoch(
                self.config,
                model,
                optimizer,
                loss_fn,
                valid_dataset,
                valid_metric,
                False,
            )

            tf.print(
                f"Epoch: {epoch+1} | time in {mins:.1f} minutes, {secs} seconds"
            )
            tf.print(
                f"\tTrainLoss: {train_loss:.4f}  |  TrainMse/Acc: {train_acc:.4f}"
            )
            tf.print(
                f"\tValidLoss: {val_loss:.4f}  |  ValidMse/Acc: {val_acc:.4f}")

    def evaluate(self):
        pass

    def test(self):
        pass

    def run_epoch(self, model, optimizer, loss_fn, dataset, metric,
                  training: bool):
        total_loss = 0.0
        steps = 0
        for step, (*inputs,
                   labels) in enumerate(dataset.batch(self.config.batch_size),
                                        start=1):
            loss, probs = self.run_step(model,
                                        optimizer,
                                        loss_fn,
                                        *inputs,
                                        labels,
                                        training=training)
            total_loss += loss.numpy()
            metric.update_state(labels, probs)
            steps += 1
        acc = metric.result()
        metric.reset_states()
        return acc, total_loss / steps

    @tf.function
    def run_step(self, model, run_model, optimizer, loss_fn, inputs, y_true,
                 training: bool):
        if training:
            with tf.GradientTape() as tape:
                loss, probs = run_model(*inputs, y_true, training)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(
                grads_and_vars=zip(grads, model.trainable_weights))
        else:
            loss, probs = run_model(*inputs, y_true, training)
        return loss, probs
