from typing import Union, List, Tuple
import tensorflow_addons as tfa
import tensor_annotations.tensorflow as ttf
from tensor_annotations.axes import Batch, Time

import tensorflow as tf
import tensorflow.keras.backend as K


class MetricStep:

    @staticmethod
    def sequence_classification(
        model,
        output: ttf.Tensor1[Batch],
        y_true: ttf.Tensor1[Batch],
    ) -> Tuple[List[Union[int, List[int]]], List[Union[int, List[int]]]]:
        """
        Parameters
        -----------
        output: The model output, most time should be the probability
        y_true: The golden labels

        Note
        ------
        Just make output and y_true to lists
        """
        y_preds = K.argmax(output, axis=-1).numpy().tolist()
        y_trues = y_true.numpy().tolist()
        return y_preds, y_trues

    @staticmethod
    def multi_label_sequance_classficiation(
        model,
        output: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time],
    ):
        ...

    @staticmethod
    def token_classification(
        model,
        output: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time],
    ) -> Tuple[List[Union[int, List[int]]], List[Union[int, List[int]]]]:
        """
        Parameters
        -----------
        output: The model output, most time should be the probability
        y_true: The golden labels

        Note
        ------
        Just make output and y_true to lists
        """
        y_preds = []
        y_trues = []
        preds = K.argmax(output, axis=-1)
        # 0 means UNK or PADDING, NOT label `O`
        text_lens = K.sum(
            tf.cast(K.not_equal(y_true, 0), dtype=tf.int32), axis=-1)

        for pred, text_len, labels in zip(preds, text_lens, y_true):
            # assume use bert tokenizer, no matter the model type
            ps = pred[:text_len].numpy().tolist()
            ts = labels[:text_len].numpy().tolist()
            # print(f"ps: {ps}")
            # print(f"ts: {ts}")
            assert len(ps) == len(
                ts), f"text_len {text_len}, length predict {len(ps)}, length true {len(ts)}"
            y_preds.append(ps)
            y_trues.append(ts)
        return y_preds, y_trues

    @staticmethod
    def token_classification_crf(
        model,
        output: ttf.Tensor2[Batch, Time],
        y_true: ttf.Tensor2[Batch, Time],
    ):
        y_preds = []
        y_trues = []
        # first element is logits
        logits = output[0]
        text_lens = K.sum(
            tf.cast(K.not_equal(y_true, 0), dtype=tf.int32), axis=-1)
        for logit, text_len, labels in zip(logits, text_lens, y_true):
            viterbi_path, _ = tfa.text.viterbi_decode(
                logit[:text_len], model.transition_params)
            ps = viterbi_path
            ts = labels[:text_len].numpy().tolist()
            # print(f"ps: {ps}")
            # print(f"ts: {ts}")
            assert len(ps) == len(
                ts), f"text_len {text_len}, length predict {len(ps)}, length true {len(ts)}"
            y_preds.append(ps)
            y_trues.append(ts)
        return y_preds, y_trues
