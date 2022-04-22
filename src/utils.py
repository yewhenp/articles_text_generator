import tensorflow as tf
import numpy as np


def sample_from(logits):
    logits, indices = tf.math.top_k(logits, k=1, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)


def sample_from_batch(logitss):
    logits, indices = tf.math.top_k(logitss, k=1, sorted=True)
    indices = indices
    return indices

